# Zone Tracker Docker Deployment

This guide covers how to deploy the Zone Tracker system using Docker containers.

## 🚀 Quick Start

### 1. Build and Run

Your database credentials are already configured in `docker-compose.yml`:

```bash
# Build and start the container
docker-compose up --build -d

# View logs
docker-compose logs -f zone-tracker

# Stop the container
docker-compose down
```

## 📋 Requirements

- **Docker** and **Docker Compose** installed
- **Network access** to your SQL Server (192.168.50.100)
- **Database credentials** with access to both FXStrat and TTG databases

## 🏗️ Architecture

The containerized system includes:

- **Zone Tracker Application** - Main processing engine
- **SQL Server ODBC Driver** - For database connectivity
- **Health Checks** - Automatic monitoring
- **Log Management** - Persistent logging with rotation
- **Resource Limits** - Memory and CPU constraints

## 📁 File Structure

```
zone_tracker/
├── Dockerfile                      # Container build instructions
├── docker-compose.yml              # Service orchestration (with credentials)
├── requirements-zone-tracker.txt   # Python dependencies (minimal)
├── README-Docker.md                # This file
├── logs/                           # Log output directory (created automatically)
└── *.py                            # Zone Tracker Python files
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_SERVER` | `192.168.50.100` | SQL Server hostname/IP |
| `DB_USERNAME` | *required* | Database username |
| `DB_PASSWORD` | *required* | Database password |
| `DB_DATABASE` | `FXStrat` | Main database name |
| `HISTODATA_DATABASE` | `TTG` | HistoData database name |

### Advanced Configuration

For different servers or credentials for each database, set:

```bash
# HistoData-specific settings (optional)
HISTODATA_SERVER=different.server.com
HISTODATA_USERNAME=different_user
HISTODATA_PASSWORD=different_password
```

## 📊 Monitoring

### Health Checks

The container includes automatic health checks every 5 minutes:

```bash
# Check container health
docker-compose ps

# View health check logs
docker inspect zone-tracker --format='{{json .State.Health}}'
```

### Logs

Logs are automatically managed with rotation:

```bash
# View real-time logs
docker-compose logs -f zone-tracker

# View specific number of lines
docker-compose logs --tail=100 zone-tracker

# Logs are also saved to ./logs/ directory
ls -la logs/
```

## 🔍 Troubleshooting

### Common Issues

**Connection refused to SQL Server:**
```bash
# Check if SQL Server is accessible from container
docker-compose exec zone-tracker ping 192.168.50.100

# Test database connection
docker-compose exec zone-tracker python -c "
import pyodbc
conn_str = 'Driver={ODBC Driver 17 for SQL Server};Server=192.168.50.100;...'
# Add your connection test here
"
```

**ODBC Driver issues:**
```bash
# List available ODBC drivers in container
docker-compose exec zone-tracker odbcinst -q -d
```

**Memory/CPU issues:**
```bash
# Check resource usage
docker stats zone-tracker

# Adjust limits in docker-compose.yml if needed
```

### Debug Mode

Run with debug output:

```bash
# Stop normal container
docker-compose down

# Run interactively for debugging
docker-compose run --rm zone-tracker python zone_tracker_main.py
```

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up --build -d

# Or use the optional Watchtower service (uncomment in docker-compose.yml)
```

### Backup Considerations

The Zone Tracker reads from HistoData and writes to FXStrat database tables. Ensure your database backup strategy covers:

- **FXStrat database** - Contains zones, indicators, trade simulation results
- **Application logs** - Located in `./logs/` directory

## 🚨 Production Deployment

For production use:

1. **Use Docker Secrets** instead of environment variables for credentials
2. **Set up monitoring** with Prometheus/Grafana
3. **Configure log aggregation** (ELK stack, Fluentd, etc.)
4. **Set resource limits** appropriate for your hardware
5. **Use a reverse proxy** (nginx) if exposing any web interfaces
6. **Set up automated backups** for logs and database

## 📞 Support

If you encounter issues:

1. Check the logs: `docker-compose logs zone-tracker`
2. Verify database connectivity
3. Ensure environment variables are set correctly
4. Check system resources (memory/CPU)

The system runs during FX market hours (Sunday 3 PM MST to Friday 3 PM MST) and processes data every 5 minutes. 