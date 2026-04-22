# Deploying To Railway

## Streamlit entrypoint

The app entrypoint is:

`app/Dashboard_Home.py`

This is the file Railway should launch with Streamlit.

## Railway start command

Use this exact start command in Railway service settings if you want to override it manually:

```bash
streamlit run app/Dashboard_Home.py --server.address 0.0.0.0 --server.port $PORT
```

The repo also includes a root `Procfile` with the same command.

## Railway environment variables

Set one of these persistence options:

1. Recommended: managed database

```env
DATABASE_URL=<your database connection string>
```

2. Simple SQLite option on a mounted Railway volume

```env
APP_DATA_DIR=/data
```

Notes:

- `PORT` is provided by Railway automatically. Do not set it manually.
- If `DATABASE_URL` is set, the app will use it.
- If `DATABASE_URL` is not set and `APP_DATA_DIR` is set, the app will store SQLite data at `/data/cmcsif_portfolio.db`.
- If neither is set, the app can still boot, but local filesystem data may be ephemeral on Railway.

## Deploy from GitHub to Railway

1. Push this repo to GitHub.
2. In Railway, create a new project and choose `Deploy from GitHub repo`.
3. Select this repository.
4. Let Railway install dependencies from `requirements.txt`.
5. In the Railway service settings, confirm the start command is:

```bash
streamlit run app/Dashboard_Home.py --server.address 0.0.0.0 --server.port $PORT
```

6. Add your environment variable choice:
   - `DATABASE_URL`, or
   - `APP_DATA_DIR=/data` if you attach a Railway volume for SQLite persistence.
7. Deploy the service.
8. Open the Railway-generated public domain first to confirm the app starts cleanly.

## Custom domain with Cloudflare

1. In Railway, open your service and go to the custom/public domain area.
2. Add your domain, for example `alpharix.com`.
3. Railway will provide:
   - a `CNAME` record target
   - a `TXT` verification record
4. In Cloudflare, create both records exactly as Railway provides.
5. Railway custom domains now require both the DNS `CNAME` record and the `TXT` verification record.
6. Wait for Railway to verify the domain and issue TLS.
7. In Cloudflare, set SSL/TLS mode to `Full`.

## Important notes

- Railway supports overriding the service start command in service settings.
- Railway variables should be used for secrets and configuration.
- Railway public networking can provide a Railway-hosted public URL before you attach your custom domain.
- Streamlit should bind to `0.0.0.0` and use Railway's assigned `$PORT`.
