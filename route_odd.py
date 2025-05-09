import matplotlib.pyplot as plt
from PIL import Image
from tum_color import TUMColor


plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})
# Load map screenshot
map_image = Image.open("data/route_without_legend.png")

# Create a figure the same size as the image
fig, ax = plt.subplots(figsize=(map_image.width / 100, map_image.height / 100), dpi=100)

# Show the map
ax.imshow(map_image)
ax.axis("off")  # Hide axes

# Example legend entries: match your geojson.io ODD layer colors
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=TUMColor.TUM_BLUE, label='Coutry Road'),
    Patch(facecolor=TUMColor.TUM_DARKRED, label='Town'),
    Patch(facecolor=TUMColor.TUM_GREEN, label='Highway'),
    Patch(facecolor=TUMColor.TUM_GRAY, alpha=0.6, label='First Route'),
    Patch(facecolor=TUMColor.TUM_PINK, alpha=0.6, label='Second Route'),
]

# Add legend
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=20)

# Save output
plt.tight_layout()
plt.savefig("out/route_odd.pdf", format="pdf", dpi=300)
