import os
import matplotlib.pyplot as plt

def pie_manual_pn_status(total, manual, text_1, text_2, text_total, name_of_file):
    # Define your data
    total_entries = total
    manual_entries = manual

    colors = ['#00b400', '#b40000']

    # Define labels for your chart
    labels = [f"{text_1}", f'{text_2}']

    # Define the amount of highlight to apply to the manual entries
    explode = [0, 0.1]

    # Create the chart
    fig1, ax1 = plt.subplots()
    patches, texts, autotexts = ax1.pie([total_entries-manual_entries, manual_entries], colors=colors, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True)

    # Set equal aspect ratio
    ax1.axis('equal')
    fig1.patch.set_alpha(0)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    # Add text label to the chart showing the total entries
    ax1.text(-1, 1.2, f'{text_total}: {total_entries}', ha='center', va='center', fontsize=14, color='green')

    
    # Save the chart as a PNG image in the "charts" directory
    filename = f'charts_images/{name_of_file}.png'
    if os.path.exists(filename):
        os.remove(filename)
    ax1.legend(labels=labels, loc='lower left')
    plt.savefig(filename, transparent=True)
