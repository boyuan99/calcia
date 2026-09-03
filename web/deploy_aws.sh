#!/usr/bin/env bash
# Deploy web/site/ to AWS as a static site: private S3 bucket + CloudFront (OAC) + HTTPS.
#
#   BUCKET=my-unique-name ./deploy_aws.sh bootstrap   once — bucket, OAC, distribution
#   ./deploy_aws.sh sync                              every time — upload + invalidate
#   ./deploy_aws.sh status                            what exists, and the live URL
#   ./deploy_aws.sh domain example.com [www.…]        ACM cert + attach to the CDN
#
# Settings land in web/.aws-deploy.env (git-ignored), written by bootstrap.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$HERE/site"
ENV_FILE="$HERE/.aws-deploy.env"

BUCKET="${BUCKET:-}"                     # globally unique, e.g. calcia-showcase-2026
REGION="${REGION:-}"                     # empty -> whatever `aws configure` has, else us-east-1
PROFILE_ARG=""
[[ -n "${AWS_PROFILE:-}" ]] && PROFILE_ARG="--profile $AWS_PROFILE"

# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

aws_() { command aws $PROFILE_ARG "$@"; }

# aws.exe is a native Windows binary, so it cannot resolve the MSYS paths that
# mktemp hands back under Git Bash. Convert before passing one as file://.
winpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

require_aws() {
  command -v aws >/dev/null 2>&1 || { echo "aws CLI not found — install it first." >&2; exit 1; }
  aws_ sts get-caller-identity >/dev/null 2>&1 || {
    echo "AWS credentials not working. Run 'aws configure' (or set AWS_PROFILE)." >&2; exit 1; }
}

# ---------------------------------------------------------------------------
bootstrap() {
  require_aws
  [[ -n "$BUCKET" ]] || { echo "Set BUCKET=<globally-unique-name> and re-run." >&2; exit 1; }
  # The bucket needs a home region; CloudFront is global and ignores this.
  if [[ -z "$REGION" ]]; then
    REGION="$(aws_ configure get region 2>/dev/null || true)"
    REGION="${REGION:-us-east-1}"
  fi
  local acct; acct="$(aws_ sts get-caller-identity --query Account --output text)"
  echo "account $acct | region $REGION | bucket $BUCKET"

  # 1. private bucket -------------------------------------------------------
  if aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "bucket exists, reusing"
  elif [[ "$REGION" == "us-east-1" ]]; then
    aws_ s3api create-bucket --bucket "$BUCKET" --region us-east-1
  else
    aws_ s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
  aws_ s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

  # 2. origin access control ------------------------------------------------
  local oac_id
  oac_id="$(aws_ cloudfront list-origin-access-controls \
      --query "OriginAccessControlList.Items[?Name=='${BUCKET}-oac'].Id | [0]" --output text)"
  if [[ "$oac_id" == "None" || -z "$oac_id" ]]; then
    oac_id="$(aws_ cloudfront create-origin-access-control \
      --origin-access-control-config "Name=${BUCKET}-oac,Description=calcia site,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
      --query 'OriginAccessControl.Id' --output text)"
  fi
  echo "oac $oac_id"

  # 3. distribution ---------------------------------------------------------
  local cfg dist_id domain dist_arn
  cfg="$(mktemp)"
  {
    printf '{\n'
    printf '  "CallerReference": "%s-%s",\n' "$BUCKET" "$(date +%s)"
    printf '  "Comment": "calcia showcase",\n'
    printf '  "Enabled": true,\n'
    printf '  "DefaultRootObject": "index.html",\n'
    printf '  "HttpVersion": "http2and3",\n'
    printf '  "PriceClass": "PriceClass_100",\n'
    printf '  "Origins": { "Quantity": 1, "Items": [{\n'
    printf '    "Id": "s3-%s",\n' "$BUCKET"
    printf '    "DomainName": "%s.s3.%s.amazonaws.com",\n' "$BUCKET" "$REGION"
    printf '    "OriginAccessControlId": "%s",\n' "$oac_id"
    printf '    "S3OriginConfig": { "OriginAccessIdentity": "" }\n'
    printf '  }] },\n'
    printf '  "DefaultCacheBehavior": {\n'
    printf '    "TargetOriginId": "s3-%s",\n' "$BUCKET"
    printf '    "ViewerProtocolPolicy": "redirect-to-https",\n'
    printf '    "AllowedMethods": { "Quantity": 2, "Items": ["GET", "HEAD"],\n'
    printf '      "CachedMethods": { "Quantity": 2, "Items": ["GET", "HEAD"] } },\n'
    printf '    "Compress": true,\n'
    printf '    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6"\n'
    printf '  },\n'
    printf '  "CustomErrorResponses": { "Quantity": 1, "Items": [{\n'
    printf '    "ErrorCode": 403, "ResponseCode": "404",\n'
    printf '    "ResponsePagePath": "/index.html", "ErrorCachingMinTTL": 60\n'
    printf '  }] }\n'
    printf '}\n'
  } > "$cfg"

  dist_id="$(aws_ cloudfront create-distribution --distribution-config "file://$(winpath "$cfg")" \
              --query 'Distribution.Id' --output text)"
  rm -f "$cfg"
  domain="$(aws_ cloudfront get-distribution --id "$dist_id" \
              --query 'Distribution.DomainName' --output text)"
  dist_arn="arn:aws:cloudfront::${acct}:distribution/${dist_id}"
  echo "distribution $dist_id -> https://$domain"

  # 4. let only that distribution read the bucket ---------------------------
  local pol
  pol="$(mktemp)"
  {
    printf '{ "Version": "2008-10-17", "Statement": [{\n'
    printf '  "Sid": "AllowCloudFrontServicePrincipal",\n'
    printf '  "Effect": "Allow",\n'
    printf '  "Principal": { "Service": "cloudfront.amazonaws.com" },\n'
    printf '  "Action": "s3:GetObject",\n'
    printf '  "Resource": "arn:aws:s3:::%s/*",\n' "$BUCKET"
    printf '  "Condition": { "StringEquals": { "AWS:SourceArn": "%s" } }\n' "$dist_arn"
    printf '}] }\n'
  } > "$pol"
  aws_ s3api put-bucket-policy --bucket "$BUCKET" --policy "file://$(winpath "$pol")"
  rm -f "$pol"

  {
    printf 'BUCKET=%s\n' "$BUCKET"
    printf 'REGION=%s\n' "$REGION"
    printf 'DIST_ID=%s\n' "$dist_id"
    printf 'DIST_DOMAIN=%s\n' "$domain"
  } > "$ENV_FILE"

  echo
  echo "wrote $ENV_FILE"
  echo "now run:  ./deploy_aws.sh sync"
  echo "the distribution needs ~5 min to go live at https://$domain"
}

# ---------------------------------------------------------------------------
# One upload pass per extension, so every object gets the right Content-Type
# and Cache-Control. Filenames carry no content hash, so HTML/CSS/JS
# revalidate on every load and only the heavy media is cached for real.
upload() {
  local ext="$1" ctype="$2" cache="$3"
  aws_ s3 cp "$SITE/" "s3://$BUCKET/" --recursive --only-show-errors \
    --exclude "*" --include "*.$ext" \
    --content-type "$ctype" --cache-control "$cache"
}

sync_() {
  require_aws
  [[ -n "${BUCKET:-}" ]] || { echo "No $ENV_FILE — run bootstrap first." >&2; exit 1; }
  [[ -d "$SITE" ]] || { echo "Missing $SITE" >&2; exit 1; }
  [[ -f "$SITE/assets/data.js" ]] || echo "warning: assets/ looks unbuilt — see web/README.md"

  local unknown
  unknown="$(cd "$SITE" && find . -type f \
    ! -name '*.html' ! -name '*.css' ! -name '*.js' ! -name '*.json' ! -name '*.bin' \
    ! -name '*.jpg' ! -name '*.png' ! -name '*.webp' ! -name '*.mp4' ! -name '*.webm')"
  if [[ -n "$unknown" ]]; then
    echo "unhandled file types (no Content-Type rule, uploaded by the delete pass):"
    echo "$unknown"
  fi

  local REV="no-cache" DAY="public, max-age=86400" WEEK="public, max-age=604800"
  upload html "text/html; charset=utf-8"       "$REV"
  upload css  "text/css; charset=utf-8"        "$REV"
  upload js   "text/javascript; charset=utf-8" "$REV"
  upload json "application/json"               "$DAY"
  upload bin  "application/octet-stream"       "$DAY"
  upload jpg  "image/jpeg"                     "$WEEK"
  upload png  "image/png"                      "$WEEK"
  upload webp "image/webp"                     "$WEEK"
  upload mp4  "video/mp4"                      "$WEEK"
  upload webm "video/webm"                     "$WEEK"

  # Everything is already uploaded, so --size-only makes this a delete-only
  # pass that drops objects no longer present in site/.
  aws_ s3 sync "$SITE/" "s3://$BUCKET/" --delete --size-only --only-show-errors

  local inv
  inv="$(aws_ cloudfront create-invalidation --distribution-id "$DIST_ID" \
          --paths '/*' --query 'Invalidation.Id' --output text)"
  echo "invalidation $inv"
  echo "live: https://$DIST_DOMAIN"
}

# ---------------------------------------------------------------------------
# Custom domain. Requests a DNS-validated ACM certificate (CloudFront only
# accepts certificates from us-east-1, whatever region the bucket lives in),
# prints the record you add at your DNS provider, then attaches the domain and
# the certificate to the distribution.
domain_() {
  require_aws
  command -v python >/dev/null 2>&1 || { echo "python needed to patch the distribution config." >&2; exit 1; }
  [[ -n "${DIST_ID:-}" ]] || { echo "No $ENV_FILE — run bootstrap first." >&2; exit 1; }
  local dom="${1:-}"
  [[ -n "$dom" ]] || { echo "usage: ./deploy_aws.sh domain example.com [www.example.com ...]" >&2; exit 1; }
  shift
  local names=("$dom" "$@")

  # 1. certificate --------------------------------------------------------
  local cert
  cert="$(aws_ acm list-certificates --region us-east-1       --query "CertificateSummaryList[?DomainName=='$dom'].CertificateArn | [0]" --output text)"
  if [[ "$cert" == "None" || -z "$cert" ]]; then
    if [[ $# -gt 0 ]]; then
      cert="$(aws_ acm request-certificate --region us-east-1 --domain-name "$dom"         --subject-alternative-names "$@" --validation-method DNS         --query CertificateArn --output text)"
    else
      cert="$(aws_ acm request-certificate --region us-east-1 --domain-name "$dom"         --validation-method DNS --query CertificateArn --output text)"
    fi
    echo "requested $cert"
  else
    echo "reusing $cert"
  fi

  # 2. the records you must add at your DNS provider ----------------------
  local tries=0 recs=""
  while [[ $tries -lt 10 ]]; do
    recs="$(aws_ acm describe-certificate --region us-east-1 --certificate-arn "$cert"       --query 'Certificate.DomainValidationOptions[].[DomainName,ResourceRecord.Name,ResourceRecord.Value]'       --output text 2>/dev/null || true)"
    [[ "$recs" == *CNAME* || "$recs" == *_* ]] && break
    tries=$((tries + 1))
  done
  echo
  echo "Add these CNAME records at your DNS provider (Cloudflare: grey cloud / DNS only):"
  echo "$recs"
  echo
  echo "Waiting for validation — this hangs until the records resolve. Ctrl-C is safe;"
  echo "re-run the same command once the records are in."
  aws_ acm wait certificate-validated --region us-east-1 --certificate-arn "$cert"
  echo "certificate issued"

  # 3. attach domain + certificate to the distribution --------------------
  local cfgfile newfile etag
  cfgfile="$(mktemp)"; newfile="$(mktemp)"
  aws_ cloudfront get-distribution-config --id "$DIST_ID" > "$cfgfile"
  etag="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['ETag'])" "$cfgfile")"
  python - "$cfgfile" "$newfile" "$cert" "${names[@]}" <<'PYPATCH'
import json, sys
src, dst, cert, *names = sys.argv[1:]
cfg = json.load(open(src))["DistributionConfig"]
cfg["Aliases"] = {"Quantity": len(names), "Items": names}
cfg["ViewerCertificate"] = {
    "ACMCertificateArn": cert,
    "SSLSupportMethod": "sni-only",
    "MinimumProtocolVersion": "TLSv1.2_2021",
}
json.dump(cfg, open(dst, "w"), indent=2)
PYPATCH
  aws_ cloudfront update-distribution --id "$DIST_ID" --if-match "$etag"     --distribution-config "file://$(winpath "$newfile")"     --query 'Distribution.Status' --output text
  rm -f "$cfgfile" "$newfile"

  echo
  echo "Now point the domain at the distribution — one CNAME per name:"
  for n in "${names[@]}"; do printf '  %-28s CNAME  %s
' "$n" "$DIST_DOMAIN"; done
  echo "(Cloudflare flattens a CNAME on the root domain, so the apex works too.)"
  echo "The distribution takes ~5 min to pick up the change."
}

status() {
  require_aws
  [[ -f "$ENV_FILE" ]] || { echo "not bootstrapped yet"; exit 0; }
  cat "$ENV_FILE"
  aws_ cloudfront get-distribution --id "$DIST_ID" \
    --query 'Distribution.{Status:Status,Domain:DomainName,Enabled:DistributionConfig.Enabled}' \
    --output table
  aws_ s3 ls "s3://$BUCKET" --recursive --summarize | tail -3
}

case "${1:-}" in
  bootstrap) bootstrap ;;
  sync)      sync_ ;;
  status)    status ;;
  domain)    shift; domain_ "$@" ;;
  *) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 1 ;;
esac
