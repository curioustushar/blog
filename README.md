# Tushar Gupta — Blog

Hugo + [Iara](https://github.com/alexandrevicenzi/iara) theme. Lives alongside the portfolio as a **project site**.

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
