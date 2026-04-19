from app.probe_texts import render_views


def test_render_returns_three_views(sample_bundle):
    views = render_views(sample_bundle)
    assert set(views.keys()) == {"summary", "work", "skills_edu"}


def test_summary_mentions_name_title_location(sample_bundle):
    views = render_views(sample_bundle)
    s = views["summary"]
    assert "Sara Ali" in s
    assert "Engineering Manager" in s
    assert "Philadelphia" in s
    assert "Latvia" in s  # nationality
    assert "16 years" in s


def test_work_mentions_company_and_dates(sample_bundle):
    views = render_views(sample_bundle)
    w = views["work"]
    assert "IDM Brokerage House" in w
    assert "Finance" in w
    assert "2025" in w


def test_skills_edu_mentions_skill_and_degree(sample_bundle):
    views = render_views(sample_bundle)
    se = views["skills_edu"]
    assert "Brand Marketing" in se
    assert "9" in se         # years for brand marketing
    assert "Associate" in se  # degree


def test_empty_sections_do_not_crash(sample_bundle):
    sample_bundle["work"] = []
    sample_bundle["education"] = []
    sample_bundle["skills"] = []
    sample_bundle["languages"] = []
    views = render_views(sample_bundle)
    assert views["summary"]              # header-only summary still produced
    assert views["work"] == "" or "no work history" in views["work"].lower()
