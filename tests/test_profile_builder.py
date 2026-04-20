from app.profile_builder import render_full_profile, render_mini


def test_render_mini_single_paragraph(sample_bundle):
    s = render_mini(sample_bundle)
    assert "Sara Ali" in s
    assert "Engineering Manager Research Scientist" in s
    assert "IDM Brokerage House" in s
    assert "Philadelphia" in s
    assert "16y" in s
    # Single paragraph — no newlines except the trailing headline
    # (acceptable: minor wrap if headline contained one)
    assert s.count("\n") <= 1


def test_render_full_profile_has_all_sections(sample_bundle):
    md = render_full_profile(sample_bundle)
    # Header
    assert "# Sara Ali" in md
    # Section headers
    assert "## Work experience" in md
    assert "## Skills" in md
    assert "## Education" in md
    assert "## Languages" in md
    # Specific data points
    assert "IDM Brokerage House" in md
    assert "Brand Marketing" in md
    assert "Union High School" in md
    assert "Italian" in md
