<#
Upload the competition data ONCE (Aryan) to s3://amlc26-<account>/raw/student_resource.tar.gz
Compressed first (~2.5 GB of TSV -> much smaller), so the upload from home internet is faster.
Usage:  powershell -ExecutionPolicy Bypass -File infra\aws\upload_data.ps1
#>
. "$PSScriptRoot\config.ps1"
$bucket = Get-BucketName
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$tgz = Join-Path $env:TEMP "student_resource.tar.gz"
if (-not (Test-Path "$root\student_resource\dataset\test\test_source1.tsv")) { throw "student_resource/ not found under $root" }
Write-Host "compressing student_resource (1-3 min) ..." -ForegroundColor Cyan
& tar -czf $tgz -C $root --exclude "__MACOSX" --exclude ".DS_Store" student_resource
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Write-Host ("archive: {0:N0} MB -> s3://$bucket/raw/" -f ((Get-Item $tgz).Length / 1MB)) -ForegroundColor Cyan
& aws s3 cp $tgz "s3://$bucket/raw/student_resource.tar.gz" --region $Region
if ($LASTEXITCODE -ne 0) { throw "upload failed (re-run to retry)" }
Remove-Item $tgz
Write-Host "uploaded. Teammates: run share_bucket.ps1 with their account ids so they can read it." -ForegroundColor Green
