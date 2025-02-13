import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_image_from_url(url):
    img = mpimg.imread(url)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def create_figure_2_1():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_1.png"
    display_image_from_url(url)

def create_figure_2_2():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_2.png"
    display_image_from_url(url)

def create_figure_2_3():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_3.png"
    display_image_from_url(url)

def create_figure_2_4():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_4.png"
    display_image_from_url(url)

def create_figure_2_5():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_5.png"
    display_image_from_url(url)

def create_figure_2_6():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_6.png"
    display_image_from_url(url)

def create_figure_2_7():
    url = "https://datascienceuwl.github.io/MTH371/figures/Chapter2/Figure_2_7.png"
    display_image_from_url(url)
