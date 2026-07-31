# Tushar Gupta — Blog

Hugo + [Iara](https://github.com/alexandrevicenzi/iara) theme. Lives alongside the portfolio as a **project site**.

## ERA V5 Submission — Mixture & Curriculum Plan

**Author:** Tushar Gupta  
**Assignment:** Mixture and Curriculum (V5)

| Artifact | Link |
|----------|------|
| **Full specification** | [docs/the-mixture-is-the-model-2-5t-pretraining-plan.md](./docs/the-mixture-is-the-model-2-5t-pretraining-plan.md) |
| **Live blog post** | https://curioustushar.github.io/blog/posts/the-mixture-is-the-model-2-5t-pretraining-plan/ |
| **Proxy experiment script + results** | [scripts/mixture_proxy_experiment.py](./scripts/mixture_proxy_experiment.py) · [scripts/mixture_proxy_results.json](./scripts/mixture_proxy_results.json) |
| **Related: 40B model design** | https://curioustushar.github.io/blog/posts/train-40b-model-design/ |
| **Related: data cleaning pipeline** | https://curioustushar.github.io/blog/posts/data-cleaning-strategies-applied/ |

**Summary:** 2.5T-token pre-training mixture across **7 capability lanes** (100% budget), Indic split 40/25/20/15 across verified/unverified/translated/synthetic tiers, protected floors (Indic ≥12%, Agentic ≥2%), **separate anneal preset** over final 125B tokens (5%), four-stage curriculum with difficulty and reasoning-length bands, **proxy experiment executed** (loader simulation + char-LM smoke run), and cleaning progress toward starved P0/P1 slots.

---

| Site | Repo | URL |
|------|------|-----|
| Portfolio (root) | [`curioustushar.github.io`](https://github.com/curioustushar/curioustushar.github.io) | https://curioustushar.github.io |
| Blog (this repo) | `-curioustushar.github.io` → rename to **`blog`** | https://curioustushar.github.io/blog/ |

## One-time deploy setup

1. **Rename this repo** to `blog`  
   [Settings → General](https://github.com/curioustushar/-curioustushar.github.io/settings) → Repository name → `blog`  
   (Required so GitHub serves it at `/blog/`, not `/-curioustushar.github.io/`.)

2. **Update your local remote** after renaming:
   ```bash
   git remote set-url origin git@github.com:curioustushar/blog.git
   ```

3. **Enable Pages** on the blog repo:  
   [Settings → Pages](https://github.com/curioustushar/blog/settings/pages) → **Source → GitHub Actions**

4. Re-run the workflow (or push any commit).

The portfolio repo (`curioustushar.github.io`) is untouched and keeps serving the root URL.

## Local development

```bash
brew install hugo   # if not installed
hugo server -D
# Open http://localhost:1313/blog/
```

## Publish a new post

1. Create `content/posts/YYYY-MM-DD-my-post.md`
2. `git add`, `git commit`, `git push`
3. GitHub Actions deploys automatically

## Structure

- `config.toml` — site config (name, menus, theme settings)
- `content/about.md` — About page
- `content/posts/` — blog posts
- `themes/iara/` — Iara theme
- `layouts/home.html` — home page layout fix for modern Hugo
