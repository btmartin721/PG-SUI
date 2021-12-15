import re
import sys
import toytree as tt
import toyplot.pdf

def get_tree_tips(tree):
    tips = re.split('[ ,\(\);]', tree)
    return([i for i in tips if i])

def basic_tree_plot(tree, out="out.pdf"):
    mystyle = {
        "edge_type": "p",
        "edge_style": {
            "stroke": tt.colors[0],
            "stroke-width": 1,
        },
        "tip_labels_align": True,
        "tip_labels_style": {"font-size": "5px"},
        "node_labels": False,
        "tip_labels": True
    }

    canvas, axes, mark = tree.draw(
        width=400,
        height=600,
        **mystyle,
    )

    toyplot.pdf.render(canvas, out)
