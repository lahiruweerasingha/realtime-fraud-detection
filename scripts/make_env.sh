#!/usr/bin/env sh
# Create .env from .env.example, replacing every `change-me` placeholder with a
# random secret. Runs once; an existing .env is never overwritten.
set -eu

if [ -f .env ]; then
    exit 0
fi

secret() {
    head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'
}

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
        *=change-me) printf '%s=%s\n' "${line%%=*}" "$(secret)" ;;
        *) printf '%s\n' "$line" ;;
    esac
done < .env.example > .env

echo "created .env with generated secrets (gitignored)"
