mkdir /app/credentials/
cat "${GOOGLE_CREDS}" | base64 -d > /app/credentials/google-creds.json
python -m cli.run_parser --s3 $PARSER_INPUT_PREFIX $EMBEDDINGS_INPUT_PREFIX --parallel
