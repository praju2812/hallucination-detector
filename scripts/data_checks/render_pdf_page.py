"""Render PDF page(s) to PNG for reading table values that the text layer drops
(bold / asterisked statistically-significant cells). READ-ONLY.

Usage:
  python render_pdf_page.py "<pdf path>" <page_1based> <out.png> [resolution] [x0 top x1 bottom]
If a crop box (pdf coords) is given, only that region is rendered.
"""
import sys, pdfplumber
pdf_path=sys.argv[1]; page=int(sys.argv[2]); out=sys.argv[3]
res=int(sys.argv[4]) if len(sys.argv)>4 else 300
pg=pdfplumber.open(pdf_path).pages[page-1]
if len(sys.argv)>=9:
    x0,top,x1,bottom=[float(x) for x in sys.argv[5:9]]
    pg=pg.crop((x0,top,x1,bottom))
pg.to_image(resolution=res).save(out)
print("saved",out)
