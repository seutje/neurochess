1. Install dependencies:
   `npm install`
2. Run the app:
   `npm run dev`
3. Train a base model to ship with the app:
   `npm run train:base`

   For custom settings (note the `--` separator):
   `npm run train:base -- --games 2 --train-sims 50 --opponent-sims 50`

   By default training resumes from the last base model in `public/models/base`.
   Use `--reset` to start fresh:
   `npm run train:base -- --reset`
