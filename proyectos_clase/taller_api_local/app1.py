from flask import Flask, request, jsonify
import sqlite3

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

app = Flask(__name__)
app.config["DEBUG"] = True # esto es lo que nos permite ver si está fallando algo

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Distant Reading Archive</h1>
    <p>This site is a prototype API for distant reading of science fiction novels.</p>
    """

@app.route('/api/v1/resources/books/all', methods=['GET'])
def get_all():
    connection = sqlite3.connect('data/books.db')
    connection.row_factory = dict_factory
    cursor = connection.cursor()
    result = cursor.execute("SELECT * FROM books").fetchall()
    connection.close()
    return jsonify({'books': result})

@app.route('/api/v1/resources/books/<string:author>', methods=['GET'])
def get_by_author(author):
    connection = sqlite3.connect('data/books.db')
    connection.row_factory = dict_factory
    cursor = connection.cursor()
    result = cursor.execute("SELECT * FROM books WHERE author=?", (author,)).fetchall()
    connection.close()
    return jsonify({'books': result})

@app.route('/api/v1/resources/books/filter', methods=['GET'])
def filter_table():
    id = request.args.get('id')
    published = request.args.get('published')
    author = request.args.get('author')

    query = "SELECT * FROM books WHERE"
    to_filter = []

    if id:
        query += " id=? AND"
        to_filter.append(id)
    if published:
        query += " published=? AND"
        to_filter.append(published)
    if author:
        query += " author=? AND"
        to_filter.append(author)

    if not (id or published or author):
        return jsonify({'error': 'No se proporcionaron parámetros'}), 400

    query = query[:-4] + ";"

    connection = sqlite3.connect('data/books.db')
    connection.row_factory = dict_factory
    cursor = connection.cursor()
    result = cursor.execute(query, to_filter).fetchall()
    connection.close()

    return jsonify({'books': result})

app.run()
