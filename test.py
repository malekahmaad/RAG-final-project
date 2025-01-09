from bs4 import BeautifulSoup

# Read the HTML file
file_path = 'test.html'  # Replace with your file path
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')
count = 0
stack = list()
stack.append(soup.body)
# print(soup.body.contents[1].get_text())
# for child in soup.body.children:
#     print("Child:", child)  # Direct child
#     if hasattr(child, 'children'):  # Check if the child can have children
#         for grandchild in child.children:
#             print("  Grandchild:", grandchild)

file_text = list()
while len(stack) > 0:
    element = stack.pop()
    if hasattr(element, 'children'):
        for child in element.children:
            stack.append(child)

    else:
        # print(element)
        file_text.append(element)

if soup.title:
    text = f"{soup.title.string}\n"
else:
    text = ""

for item in file_text[::-1]:
    text += item

print(text)