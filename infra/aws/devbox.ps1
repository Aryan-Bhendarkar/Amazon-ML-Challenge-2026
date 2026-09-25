<#
Manage your cloud dev box (EC2) and keep VS Code's SSH config in sync.
  devbox.ps1 up     [-Type r7i.xlarge] [-Name box]  launch (first time) - Ubuntu 24.04, 150 GB gp3, bootstrap
  devbox.ps1 up     -Gpu [-Type g6.xlarge] -Name gpu   GPU box (NVIDIA driver AMI); needs G/VT quota >= 4
  devbox.ps1 start|stop|status|ip [-Name box]        (ip = re-allow your current IP + refresh ssh config)
  devbox.ps1 resize -Type r7i.2xlarge [-Name box]    stop -> change type -> start (after quota approval)
  devbox.ps1 down   [-Name box]                      TERMINATE (deletes the disk!) - asks for confirmation
After `up`/`start`: VS Code -> Remote-SSH -> host  amlc-<Name>   (or: ssh amlc-<Name>)
#>
param([Parameter(Position = 0, Mandatory = $true)][ValidateSet("up", "start", "stop", "status", "ip", "resize", "down")][string]$Action,
      [string]$Name = "box", [string]$Type = "", [switch]$Gpu)
. "$PSScriptRoot\config.ps1"
$Tag = "$Prefix-$Member-$Name"
$HostAlias = "amlc-$Name"

function Get-Box {
    $id = Invoke-Aws ec2 describe-instances --filters "Name=tag:Name,Values=$Tag" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "Reservations[].Instances[].InstanceId | [0]" --output text
    if ($id -eq "None" -or -not $id) { return $null } else { return $id }
}

function Update-SshConfig($ip) {
    New-Item -ItemType Directory -Force (Split-Path $SshConfig) | Out-Null
    $block = @"
Host $HostAlias
    HostName $ip
    User ubuntu
    IdentityFile $($KeyFile -replace '\\','/')
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 10
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ~/.ssh/known_hosts_amlc
"@
    $txt = if (Test-Path $SshConfig) { Get-Content $SshConfig -Raw } else { "" }
    $pattern = "(?ms)^Host $([regex]::Escape($HostAlias))\r?\n.*?(?=^Host |\z)"
    if ($txt -match $pattern) { $txt = [regex]::Replace($txt, $pattern, $block + "`n") } else { $txt = $txt.TrimEnd() + "`n`n" + $block + "`n" }
    [IO.File]::WriteAllText($SshConfig, $txt.TrimStart())
    Write-Host "ssh config: Host $HostAlias -> $ip" -ForegroundColor Green
}

function Allow-MyIp {
    $sg = Invoke-Aws ec2 describe-security-groups --filters "Name=group-name,Values=$SgName" --query "SecurityGroups[0].GroupId" --output text
    $ip = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim()
    & aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$ip/32" --region $Region 2>$null | Out-Null
    Write-Host "SSH allowed from $ip/32"
}

function Wait-Running($id) {
    Invoke-Aws ec2 wait instance-running --instance-ids $id | Out-Null
    $ip = Invoke-Aws ec2 describe-instances --instance-ids $id --query "Reservations[0].Instances[0].PublicIpAddress" --output text
    Update-SshConfig $ip
    return $ip
}

$id = Get-Box
switch ($Action) {
    "up" {
        if ($id) { Write-Host "$Tag already exists ($id). Use start/status." ; break }
        if (-not $Type) { $Type = if ($Gpu) { "g6.xlarge" } else { "r7i.xlarge" } }
        if ($Gpu) {
            $ami = $null
            foreach ($p in "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id",
                           "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id") {
                $ami = & aws ssm get-parameter --name $p --region $Region --query "Parameter.Value" --output text 2>$null
                if ($LASTEXITCODE -eq 0 -and $ami) { break }
            }
        } else {
            $ami = Invoke-Aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id" --query "Parameter.Value" --output text
        }
        $sg = Invoke-Aws ec2 describe-security-groups --filters "Name=group-name,Values=$SgName" --query "SecurityGroups[0].GroupId" --output text
        $bucket = Get-BucketName
        $boxSetup = (Get-Content "$PSScriptRoot\box_setup.sh" -Raw) -replace "`r", ""
        $ud = ((Get-Content "$PSScriptRoot\user_data.sh" -Raw) -replace "`r", "").Replace("__AMLC_BUCKET__", $bucket).Replace("__BOX_SETUP__", $boxSetup.TrimEnd())
        $udFile = New-TemporaryFile; [IO.File]::WriteAllText($udFile, $ud)
        $bdm = "[{`"DeviceName`":`"/dev/sda1`",`"Ebs`":{`"VolumeSize`":$DiskGb,`"VolumeType`":`"gp3`",`"DeleteOnTermination`":true}}]"
        $bdmFile = New-TemporaryFile; [IO.File]::WriteAllText($bdmFile, $bdm)
        Write-Host "launching $Type ($ami) as $Tag ..." -ForegroundColor Cyan
        $id = Invoke-Aws ec2 run-instances --image-id $ami --instance-type $Type --key-name $KeyName `
            --security-group-ids $sg --iam-instance-profile "Name=$ProfileName" `
            --block-device-mappings "file://$bdmFile" --user-data "file://$udFile" `
            --instance-initiated-shutdown-behavior stop `
            --metadata-options "HttpTokens=required,HttpEndpoint=enabled" `
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$Tag},{Key=Project,Value=$Prefix}]" "ResourceType=volume,Tags=[{Key=Name,Value=$Tag},{Key=Project,Value=$Prefix}]" `
            --query "Instances[0].InstanceId" --output text
        $ip = Wait-Running $id
        Write-Host "$Tag running: $id  $ip. Bootstrap takes ~4 min (check: ssh $HostAlias 'cat /var/log/amlc-bootstrap.done')" -ForegroundColor Green
    }
    "start" { if (-not $id) { throw "no box $Tag - run: devbox.ps1 up" }; Allow-MyIp; Invoke-Aws ec2 start-instances --instance-ids $id | Out-Null; $ip = Wait-Running $id; Write-Host "started $id $ip" }
    "stop" { if ($id) { Invoke-Aws ec2 stop-instances --instance-ids $id | Out-Null; Write-Host "stopping $id (disk kept; only EBS ~`$0.45/day billed while stopped)" } }
    "status" {
        if (-not $id) { Write-Host "no box $Tag"; break }
        Invoke-Aws ec2 describe-instances --instance-ids $id --query "Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]" --output table
    }
    "ip" { Allow-MyIp; if ($id) { $ip = Invoke-Aws ec2 describe-instances --instance-ids $id --query "Reservations[0].Instances[0].PublicIpAddress" --output text; if ($ip -ne "None") { Update-SshConfig $ip } } }
    "resize" {
        if (-not $id -or -not $Type) { throw "need an existing box and -Type" }
        Invoke-Aws ec2 stop-instances --instance-ids $id | Out-Null; Invoke-Aws ec2 wait instance-stopped --instance-ids $id | Out-Null
        Invoke-Aws ec2 modify-instance-attribute --instance-id $id --instance-type "Value=$Type" | Out-Null
        Invoke-Aws ec2 start-instances --instance-ids $id | Out-Null; $ip = Wait-Running $id; Write-Host "resized to $Type, $ip"
    }
    "down" {
        if (-not $id) { break }
        $ok = Read-Host "TERMINATE $Tag ($id) and DELETE its disk? type yes"
        if ($ok -eq "yes") { Invoke-Aws ec2 terminate-instances --instance-ids $id | Out-Null; Write-Host "terminating $id" }
    }
}
