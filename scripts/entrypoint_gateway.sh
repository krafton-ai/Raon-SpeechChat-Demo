#!/bin/bash
set -e

CERT_DIR="/app/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

GATEWAY_PROTO="${GATEWAY_PROTO:-https}"
GATEWAY_EXTRA_ARGS=""

if [ "$GATEWAY_PROTO" = "https" ]; then
    # Auto-generate self-signed certs if missing
    if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
        mkdir -p "$CERT_DIR"
        openssl req -x509 -newkey rsa:2048 \
            -keyout "$KEY_FILE" \
            -out "$CERT_FILE" \
            -days 365 -nodes \
            -subj "/CN=raon-speechchat-demo" \
            2>/dev/null
        echo "[Certs] Auto-generated self-signed TLS certs (valid 365 days)"
    else
        echo "[Certs] Using existing TLS certs"
    fi
    GATEWAY_EXTRA_ARGS="--ssl-certfile $CERT_FILE --ssl-keyfile $KEY_FILE"
else
    GATEWAY_EXTRA_ARGS="--http"
fi

exec python3 /app/Raon-SpeechChat-Demo/launch_gateway.py "$@" $GATEWAY_EXTRA_ARGS
