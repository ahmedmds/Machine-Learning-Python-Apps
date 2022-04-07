import hug
import json

# Load data
with open("books.json") as f:
    books_data = json.load(f)


# Expose as CLI
# To run, open Terminal and activate environment:
# (venv) ...\Hug_intro>hug -f app.py -c help
# (venv) ...\Hug_intro>hug -f app.py -c get_book_by_title "Things Fall Apart"
@hug.cli()

# Expose as API
# To run, open Terminal and:
# (venv) ...\Hug_intro>hug -f app.py
@hug.get('/api/v1/books')
def get_books():
    """Show all books"""
    return{"results": books_data}

@hug.cli()
@hug.get('/api/v1/books/searchtitle', examples="title=Things Fall Apart")
def get_book_by_title(title:hug.types.text):
    """Search book by title"""
    book = [book for book in books_data if book["title"]==title]
    return {"results": book}

@hug.cli()
@hug.get('/api/v1/books/searchlanguage', examples="language=English")
def get_book_by_language(language:hug.types.text):
    """Search book by language"""
    book = [book for book in books_data if book["language"]==language]
    return {"results": book}

# For CLI only
if __name__ == '__main__':
	get_books.interface.cli()