# Operations Runbook

## Deployment

The deployment pipeline is defined in [ci.yml](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/.github/workflows/ci.yml).

Backend deploy flow:

1. Build and push backend image to ECR
2. Run Alembic migrations as an ECS task
3. Force a new ECS deployment
4. Run smoke tests against `/livez`, `/readyz`, and `/api/ml/status`

Frontend deploy flow:

1. Build the React app
2. Upload static assets to the frontend S3 bucket
3. Invalidate CloudFront
4. Optionally run a smoke test against `FRONTEND_URL`

Lambda deploy flow:

1. Zip [handler.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/lambda/ingestion/handler.py)
2. Update the ingestion Lambda function code

## Health Endpoints

- [main.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/main.py)

Endpoints:

- `/livez`: process liveness
- `/readyz`: dependency readiness
- `/health`: detailed dependency health payload
- `/metrics`: Prometheus metrics endpoint

## Logs and Monitoring

Terraform monitoring resources:

- [cloudwatch.tf](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/infrastructure/terraform/cloudwatch.tf)

Important signals:

- ECS CPU / memory alarms
- API Gateway 5xx / latency alarms
- ALB unhealthy host count
- RDS CPU
- ingestion Lambda errors / missed schedule
- AWS budget threshold alert

Application metrics:

- [metrics.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/observability/metrics.py)

## Common Issues

### Backend is up but not ready

Check:

- `/readyz`
- model load status from `/api/ml/status`
- database connectivity
- recent ECS task logs in CloudWatch

### Frontend shows stale assets

Check:

- latest CloudFront invalidation
- S3 upload timestamps
- cache headers on `index.html`

### Ingestion did not run

Check:

- EventBridge rule
- Lambda invocation history
- Lambda DLQ
- CloudWatch alarm for ingestion SLA

### Scout or summary streaming fails

Check:

- Anthropic API configuration
- backend logs from the streaming endpoints
- Redis availability if using pub/sub fan-out

## Rollback

Model rollback:

- use the model rollback API backed by S3 artifact versions

Application rollback:

- redeploy a prior ECR image tag through the ECS service

Infrastructure rollback:

- use Terraform plan/apply against the prior known-good configuration
