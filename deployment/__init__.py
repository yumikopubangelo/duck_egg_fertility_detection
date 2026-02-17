"""
Deployment configuration package for duck egg fertility detection.

This package contains all necessary configuration files for deploying the duck egg fertility detection application in production environments.

## Deployment Architecture

The application uses a multi-container Docker setup with the following services:

- **web**: Main Flask application container
- **db**: PostgreSQL database container
- **redis**: Redis cache container for background tasks
- **nginx**: Reverse proxy and static file server

## Quick Start

1. Build and start all services:
   ```bash
   docker-compose up -d --build
   ```

2. View logs:
   ```bash
   docker-compose logs -f
   ```

3. Stop services:
   ```bash
   docker-compose down
   ```

## Configuration Files

- `Dockerfile`: Multi-stage build for the web application
- `docker-compose.yml`: Service orchestration and networking
- `nginx.conf`: Reverse proxy and static file serving configuration
- `requirements-deploy.txt`: Production dependencies

## Environment Variables

Create a `.env` file with the following variables:

```
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
```

## Port Mapping

- **80**: Nginx (public access)
- **5000**: Flask application (internal)
- **5432**: PostgreSQL (internal)
- **6379**: Redis (internal)

## Data Persistence

- Database data is persisted in a Docker volume
- Uploaded files and models are persisted in bind mounts
- Static files are served by Nginx from the host filesystem

## Health Checks

The application includes health checks for all services:
- Web application: HTTP endpoint at /health
- Database: Connection validation
- Redis: Connection validation

## Scaling

To scale the web application:
```bash
docker-compose up -d --scale web=3
```

## Security

- Non-root user in Docker container
- Environment variables for secrets
- HTTPS support (configure SSL certificates in nginx.conf)
- Input validation and authentication middleware

## Monitoring

- Health check endpoints
- Prometheus metrics (if enabled)
- Log aggregation (configure external logging)

## Backup and Restore

Database backup:
```bash
docker-compose exec db pg_dump -U duck duckdb > backup.sql
```

Database restore:
```bash
docker-compose exec -T db psql -U duck duckdb < backup.sql
```

## Development vs Production

For development, use:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## Troubleshooting

Common issues and solutions:
- Port conflicts: Check if ports 80, 5000, 5432, 6379 are in use
- Database connection: Verify environment variables and network
- File permissions: Ensure proper ownership of data directories

## License

This deployment configuration is part of the duck egg fertility detection project.
"""
