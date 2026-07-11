# ChinaXiv Sample Layout Comparison

Sample file: `chinaxiv-202509.00064.pdf`

Target file: `paper/paper_chinaxiv_final_with_bottom_line.pdf`

## Measured Layout

| Metric | Sample | Target |
| --- | ---: | ---: |
| pages | 18 | 26 |
| page_width_pt | 612.0 | 612.0 |
| page_height_pt | 792.0 | 792.0 |
| left_margin_pt | 132.8 | 133.8 |
| right_margin_pt | 132.8 | 132.2 |
| top_text_pt | 133.0 | 124.9 |
| bottom_text_pt | 87.6 | 78.6 |

## Font Observations

- Sample common font sizes: `[(10.0, 844), (9.0, 18), (12.0, 18), (7.0, 13), (14.3, 9), (7.1, 1)]`
- Target common font sizes: `[(9.9, 807), (8.1, 6), (12.0, 4), (14.1, 3)]`
- Sample common fonts: `[('LMRoman10-Regular', 816), ('LMRoman10-Bold', 28), ('LMRoman12-Bold', 27), ('LMRoman9-Regular', 18), ('LatinModernMath-Regular', 13)]`
- Target common fonts: `[('TimesNewRomanPSMT', 813), ('TimesNewRomanPS-BoldMT', 7)]`

## Formatting Decision

The target manuscript uses Letter page size, narrow centered text width, 10 pt body text, 12 pt section headings, a bottom footer rule on every page, academic figure/table captions, numbered citations, and numbered references to match the observable ChinaXiv sample style as closely as practical in Word format.
