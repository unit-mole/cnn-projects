# Cloudflare Pages deployment

## Before deployment

1. Train the model.
2. Run `python scripts/convert_to_tfjs.py --model-path models/alexnet_cifar10.keras`.
3. Confirm `web/tfjs_model/model.json` and every referenced `.bin` shard exist.
4. Run `python scripts/validate_project.py`.
5. Test locally with `python scripts/run_local_web_server.py`.

## Git integration for this monorepo

1. Sign in to Cloudflare.
2. Open **Workers & Pages**.
3. Create a Pages application and connect the GitHub repository.
4. Select the `cnn-projects` repository.
5. Set the root directory to `07-image-classification-alexnet-transfer-learning`.
6. Select no framework preset.
7. Leave the build command blank for this static site.
8. Set the output directory to `web`.
9. Deploy.
10. Open the generated `pages.dev` URL and verify model loading, image upload, sample images, predictions, and responsible-use text.

## Wrangler deployment

```bash
npm install
npx wrangler login
npm run deploy
```

`wrangler.toml` sets `pages_build_output_dir = "./web"`.

## Troubleshooting

### Model fetch fails

- Use a local HTTP server; do not open `index.html` directly with `file://`.
- Confirm paths in `model.json` are relative and the shard files are in the same directory.
- Confirm capitalization matches exactly.
- Inspect the browser Network tab for 404 responses.

### Model is too large

- keep the compact global-average-pooling head,
- reduce dense layer width,
- use converter quantization after validating accuracy,
- split weights into shards,
- use Git LFS where appropriate,
- publish a smaller browser demo model and keep the full research model outside the static site.

### Deployment updates do not appear

- check the production branch,
- inspect the latest Pages deployment log,
- hard-refresh the browser,
- verify that the correct monorepo root and output directory were selected.
