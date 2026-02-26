# CHANGELOG

## 2026-02-26

### Major improvements
- Reworked app into **RAG Content Studio** with modern UI polish.
- Added **Light/Dark theme toggle** with vibrant color system.
- Introduced **Fast Mode** for low-latency content generation.

### Retrieval & generation
- Added extractive summary path with:
  - sentence deduplication
  - better sentence ranking
  - structured output format
  - chunk-level source references
- Added fast content writers for:
  - Blog intro
  - LinkedIn post (catchy structure + hashtags)
- Improved prompt-grounding behavior and output cleaning.

### Image pipeline
- Added faster image defaults for low-end machines.
- Added prompt-only fast image mode toggle.
- Improved direct image generation reliability with fallback behavior.

### LinkedIn integration
- Added `linkedin_integration.py` for OAuth + posting flow.
- Added `.env`-based config loading.
- Implemented sidebar-based connect/post UX for reliability.
- Fixed multiple posting issues and switched to stable endpoint:
  - `POST /v2/ugcPosts`

### UX & stability fixes
- Fixed Windows `rag_store` replacement permission issues.
- Reduced noisy model logs and improved model caching.
- Improved uploader/background/content card theming consistency.

### Documentation updates
- Rewrote `README.md` to reflect current architecture.
- Updated `PROJECT_OVERVIEW.md` to latest implementation.
- Updated `RESUME_DESCRIPTION.md` with polished, current summary.
- Cleaned `requirements_rag.txt` to match active dependencies.

### Cleanup
- Removed unnecessary eval/temp/cache artifacts.
