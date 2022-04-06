import hug

# Expose as local package
# To run, open Terminal and then Python interpreter:
# >>> from app import get_books
# >>> get_books("Book Name")
# {'title': 'BOOK NAME'}
# >>> exit()
@hug.local()
def get_books(title:hug.types.text):
    """Get book by title"""
    return{"title": title.upper()}

# Expose as API
# To run, open Terminal and:
# (venv) ...\Hug_intro>hug -f app.py
@hug.get('/books', examples="title=studyguide")
@hug.local()
def get_books(title:hug.types.text):
    """Get book by title"""
    return{"title": title.upper()}

# Expose as CLI
# To run, open Terminal and activate environment:
# (venv) ...\Hug_intro>hug -f app.py -c help
# (venv) ...\Hug_intro>hug -f app.py -c get_books "Book Name"
@hug.cli()
@hug.get('/books', examples="title=studyguide")
@hug.local()
def get_books(title:hug.types.text):
    """Get book by title"""
    return{"title": title.upper()}

# For CLI only
if __name__ == '__main__':
	get_books.interface.cli()
