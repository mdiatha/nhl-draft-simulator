# Operations Runbook

## Deployment

The deployment pipeline is defined in [ci.yml](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/.github/workflows/ci.yml).

Backend deploy flow:

1. Build and push backend image to ECR (tagged with `github.sha` + `latest`)
2. Deploy via SSM Run Command to the EC2 instance
3. On the instance: ECR login, `docker compose pull api`, prune old images
4. Run Alembic migrations (`docker compose run --rm api alembic upgrade head`)
5. Restart API container (`docker compose up -d --no-deps api`)
6. Run smoke tests against `/livez`, `/readyz`, and `/api/ml/status`

Frontend deploy flow:

1. Build the React app (`npm run build`)
2. Upload static assets to the frontend S3 bucket (hashed assets: immutable cache, `index.html`: no-cache)
3. Invalidate CloudFront (`/*`)
4. Optionally run a smoke test against `FRONTEND_URL`

Lambda deploy flow:

1. Zip [handler.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/lambda/ingestion/handler.py)
2. Update the ingestion Lambda function code

## Health Endpoints

- [main.py](/Users/mihirdiatha/Desktop/Projects/NHL draft project/nhl-draft-simulator/backend/app/main.py)

Endpoints:

- `/livez`: process liveness — always 200 if the process is running
- `/readyz`: dependency readiness — 503 until DB is connected and model is loaded
- `/health`: detailed dependency health payload (DB status, model loaded, prospect/team counts)

## Logs and Monitoring

Application logs are written as structured JSON via `app/observability/logging.py`. Each log line includes a `request_id` field for tracing a request across log entries.

To view live logs on the instance:

```bash
# Via SSM Session Manager (no SSH key needed)
aws ssm start-session --target <EC2_INSTANCE_ID>

# Then on the instance:
docker compose logs -f api
docker compose logs -f db
```

Important signals to watch:

- EC2 CPU and memory (CloudWatch EC2 metrics)
- Docker Compose service health (`docker compose ps`)
- EventBridge ingestion rule execution history
- Lambda invocation errors / missed schedule
- AWS Budget threshold alert (SNS email at configured threshold)

## Common Issues

### Backend is up but not ready

Check:

- `/readyz`
- model load status from `/api/ml/status`
- database connectivity
- recent container logs: `docker compose logs api --tail=100`

### Frontend shows stale assets

Check:

- latest CloudFront invalidation
- S3 upload timestamps
- cache headers on `index.html`

### Ingestion did not run

Check:

- EventBridge rule enabled/disabled status
- Lambda invocation history in CloudWatch Logs
- Lambda DLQ (if configured)

### Scout or summary streaming fails

Check:

- Anthropic API key is set (`ANTHROPIC_API_KEY` in `.env`)
- Ollama is running: `docker compose ps ollama`
- Backend logs from the streaming endpoints

## Rollback

Model rollback:

- Use `POST /api/ml/rollback?version=<trained_at>` to copy a prior S3 artifact back to `models/latest/` and hot-reload the registry

Application rollback:

- Set `IMAGE_TAG` in `/home/ec2-user/.env` to a prior ECR image SHA, then `docker compose up -d --no-deps api`
- Or push a prior git tag to trigger a full CI/CD redeploy

Infrastructure rollback:

- Run `npx cdk deploy` against a prior commit of `infrastructure/cdk/lib/nhl-draft-stack.ts`
