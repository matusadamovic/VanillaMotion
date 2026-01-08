## Deploy
- Bot: `uvicorn bot.app:app --host 0.0.0.0 --port 8000` a nastav webhook `https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook?url=<bot-url>/webhook`.
- DB migracia: `psql $DATABASE_URL -f migrations/001_init.sql`.
- Worker image: `docker build -t imgvidbot-worker -f Dockerfile.worker .` a pushni do registry.
- RunPod endpoint: pouzi `infra/runpod-endpoint.json` ako sablonu, nastav secrets a obrazok.

## Rotacia secretov
- Vygeneruj nove tokeny (Telegram, RunPod, HF, DB).
- Zmen env v bot sluzbe a RunPod endpointe, re-deploy.
- Revoke stare tokeny.

## Zvysenie workersMax
- Upravit endpoint `maxWorkers` na pozadovane cislo (2+), sledovat DB load.
- Ak treba, pridat frontu/rate-limit.

## Incidenty
- Zabitie workeru: RunPod job sa zopakuje, runner skontroluje stav a edituje povodny placeholder, z /tmp sa zmaze po boote.
- Faily uploadu do Telegramu: runner fallbackne na `sendVideo`.
- Cache miss modelov: worker zlyha, stav v DB = FAILED; oprav HF cache a opakuj.

## Overenie
- Health: `GET /healthz` bot; DB `SELECT 1`.
- Rychly end-to-end: posli foto v sandbox chate, ocakavaj placeholder + finalne video v ~5-10 min.

## Cistota dat
- Ziadne binarne data mimo /tmp; runner aj bootstrap robia best-effort cleanup.
- DB drzi iba metadate, ziadne bajty.
