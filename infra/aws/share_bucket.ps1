<#
Let teammates' AWS accounts READ your bucket (data + model artifacts), per the challenge guide.
Usage:  powershell -ExecutionPolicy Bypass -File infra\aws\share_bucket.ps1 -Accounts 111122223333,444455556666
Teammates then set  AMLC_DATA_BUCKET=<this bucket>  on their box (their instance role already allows amlc26-* buckets).
#>
param([Parameter(Mandatory = $true)][string[]]$Accounts)
. "$PSScriptRoot\config.ps1"
$bucket = Get-BucketName
$principals = ($Accounts | ForEach-Object { "`"arn:aws:iam::$($_.Trim()):root`"" }) -join ","
$policy = "{`"Version`":`"2012-10-17`",`"Statement`":[{`"Sid`":`"TeamRead`",`"Effect`":`"Allow`",`"Principal`":{`"AWS`":[$principals]},`"Action`":[`"s3:GetObject`",`"s3:ListBucket`"],`"Resource`":[`"arn:aws:s3:::$bucket`",`"arn:aws:s3:::$bucket/*`"]}]}"
$f = [IO.Path]::GetTempFileName(); Set-Content $f $policy -Encoding ascii
Invoke-Aws s3api put-bucket-policy --bucket $bucket --policy "file://$f" | Out-Null
Write-Host "s3://$bucket is readable by: $($Accounts -join ', ')" -ForegroundColor Green
Write-Host "Tell teammates:  AMLC_DATA_BUCKET=$bucket bash ~/box_setup.sh"
