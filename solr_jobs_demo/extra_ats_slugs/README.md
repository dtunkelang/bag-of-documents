# Extra ATS tenant slugs

Greenhouse / Lever / Ashby company slugs to poll **in addition to** OpenApply's
`cc_*_FINAL.txt` lists. Those FINAL lists (~15k tenants, harvested from Common
Crawl) already cover essentially every mainstream company — a probe of ~190
notable firms found only a handful missing — so this is a **targeted hook for
specific companies the harvest missed**, not a bulk discovery source.

## Usage

One slug per line in the matching file. The slug is the tenant id in the ATS URL:

| ATS        | URL                                            | slug        | file                    |
|------------|------------------------------------------------|-------------|-------------------------|
| Greenhouse | `boards.greenhouse.io/<slug>`                  | `<slug>`    | `cc_greenhouse_EXTRA.txt` |
| Lever      | `jobs.lever.co/<slug>`                         | `<slug>`    | `cc_lever_EXTRA.txt`      |
| Ashby      | `jobs.ashbyhq.com/<slug>`                      | `<slug>`    | `cc_ashby_EXTRA.txt`      |

Lines starting with `#` are ignored. Any slug already present in the
corresponding `cc_*_FINAL.txt` is automatically skipped (it's covered by the
main crawl), so duplicates here are harmless.

`refresh.py` stage 0 polls these via the same `oa_adapter.py` used by the main
crawl, builds the `jobs_data_ats_extra` corpus, and unify merges it. Disable with
`--skip-ats-extra`. To find a candidate's slug, open its careers page and read
the ATS URL; confirm it returns jobs, e.g.
`curl -s https://boards-api.greenhouse.io/v1/boards/<slug>/jobs | head`.
