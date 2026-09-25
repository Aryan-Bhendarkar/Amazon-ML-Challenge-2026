#!/bin/bash
# Aryan only: publish a read-only code snapshot (committed HEAD, no data/secrets) for teammates.
#   bash infra/aws/publish_snapshot.sh
set -euo pipefail
cd ~/amlc
source /etc/profile.d/amlc.sh
SHA=$(git rev-parse --short HEAD)
git archive --format=tar.gz -o /tmp/amlc-$SHA.tar.gz HEAD
aws s3 cp /tmp/amlc-$SHA.tar.gz "s3://$AMLC_BUCKET/share/code/amlc-$SHA.tar.gz" --only-show-errors
aws s3 cp /tmp/amlc-$SHA.tar.gz "s3://$AMLC_BUCKET/share/code/amlc-latest.tar.gz" --only-show-errors
echo "published snapshot $SHA ($(du -h /tmp/amlc-$SHA.tar.gz | cut -f1)) -> s3://$AMLC_BUCKET/share/code/amlc-latest.tar.gz"
