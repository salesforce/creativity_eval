from flask import Flask, jsonify, render_template, request, redirect, url_for, session, make_response
import os
import json
import difflib
import openai
import threading
from collections import defaultdict

openai.api_key = 'sk-proj-cBs3XbtEoUXzWIslBalRT3BlbkFJgRqyiSp9eDVbQgkihoPo'

app = Flask(__name__)
app.secret_key = os.urandom(24)
whitelisted = ['issa', 'stefan','alicia','imani','teddy','jameson','thomas','margaret','erin','sophia','gene','brianna','rachelle','riley','rachel_lapides','xiadi','yen','riley','hyo','tuhin','cgreer','adeniyiademoroti','micah']

# Dictionary to store user-specific locks
user_locks = defaultdict(threading.Lock)

def get_user_lock(user_name):
    return user_locks[user_name]

def get_user_data(user_name):
    with get_user_lock(user_name):
        with open(f'{user_name}.json') as f:
            schema = json.load(f)

        completed_pages = set()
        if os.path.exists(f'./{user_name}_creativity_scores.json'):
            with open(f'./{user_name}_creativity_scores.json') as f:
                creativity_scores = json.load(f)
            for score in creativity_scores:
                completed_pages.add(score['page_id'])

        data = {}
        for i, item in enumerate(schema):
            iden = i + 1
            if item['well-formed'] == 'True':
                data[iden] = {
                    "id": iden,
                    "instruction": item['instruction'],
                    "paragraph": item['generated_text'],
                    "well-formed": item['well-formed'],
                    "completed": iden in completed_pages
                }
        return data, completed_pages

def get_last_completed_page_id(user_name):
    with get_user_lock(user_name):
        if os.path.exists(f'./{user_name}_creativity_scores.json'):
            with open(f'./{user_name}_creativity_scores.json') as f:
                creativity_scores = json.load(f)
            if creativity_scores:
                return max(score['page_id'] for score in creativity_scores)
    return 0

@app.route('/view_edits/<user_name>', methods=['GET'])
def view_edits(user_name):
    if user_name not in whitelisted:
        return jsonify({"error": "User not authorized"}), 403

    edit_file = f'{user_name}_edit.json'

    with get_user_lock(user_name):
        if os.path.exists(edit_file):
            with open(edit_file, 'r') as file:
                edits = json.load(file)
            file.close()
            return jsonify(edits)
        else:
            return jsonify({"message": "No edits found for this user"}), 404

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        full_name = request.form['fullName']
        user_name = request.form['userName']
        if user_name not in whitelisted:
            return render_template('login.html', error="User not whitelisted")
        session['full_name'] = full_name
        session['user_name'] = user_name
        last_completed_page_id = get_last_completed_page_id(user_name)
        next_page_id = last_completed_page_id + 1
        return redirect(url_for('page', page_id=next_page_id))
    return render_template('login.html')

@app.route('/')
def home():
    resp = {'message': 'Login here at https://creative-writing-edits.salesforceresearch.ai/login', 'code': 'SUCCESS'}
    return make_response(jsonify(resp), 200)

def highlight_diff(old_text, new_text):
    d = difflib.Differ()
    diff = list(d.compare(old_text.split(), new_text.split()))

    result = []
    for word in diff:
        if word.startswith('  '):
            result.append(word[2:])
        elif word.startswith('- '):
            result.append(f'<span style="background-color: #ffcccc;">{word[2:]}</span>')
        elif word.startswith('+ '):
            result.append(f'<span style="background-color: #ccffcc;">{word[2:]}</span>')

    return ' '.join(result)

@app.route('/visualize_edits', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        old_paragraph = request.form['old_paragraph']
        new_paragraph = request.form['new_paragraph']
        highlighted_diff = highlight_diff(old_paragraph, new_paragraph)
        return render_template('result.html', diff=highlighted_diff)
    return render_template('visualize.html')

@app.route('/undo_last_edit', methods=['POST'])
def undo_last_edit():
    user_name = session.get('user_name')
    if not user_name:
        return jsonify({"status": "error", "message": "User not logged in"})

    edit_file = f'{user_name}_edit.json'

    with get_user_lock(user_name):
        if os.path.exists(edit_file):
            with open(edit_file, 'r') as file:
                edits = json.load(file)
            if edits:
                last_edit = edits.pop()  # Remove the last edit

                with open(edit_file, 'w') as file:
                    json.dump(edits, file, indent=4)

                return jsonify({"status": "success", "message": "Last edit undone", "lastEdit": last_edit})
    return jsonify({"status": "error", "message": "No edits to undo"})

def save_to_json(data, filename):
    """Function to save data to a JSON file."""
    user_name = session.get('user_name')
    if not user_name:
        return

    filename = f'{user_name}_{filename}'
    with get_user_lock(user_name):
        if os.path.exists(filename):
            with open(filename, 'r+') as file:
                file_data = json.load(file)
                file_data.append(data)
                file.seek(0)
                json.dump(file_data, file, indent=4)
        else:
            with open(filename, 'w') as file:
                json.dump([data], file, indent=4)

@app.route('/page/<int:page_id>', methods=['GET', 'POST'])
def page(page_id):
    if 'full_name' not in session or 'user_name' not in session:
        return redirect(url_for('login'))

    user_name = session['user_name']
    data, completed_pages = get_user_data(user_name)

    if page_id not in data:
        return "Page not found!", 404

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'save_edit':
            edit_data = {
                'page_id': page_id,
                'categorization': request.form.get('categorization'),
                'originalText': request.form['originalText'],
                'editedText': request.form['editedText'],
                'fullTextPreEdit': request.form['fullTextPreEdit'],
                'fullTextPostEdit': request.form['fullTextPostEdit']
            }
            if edit_data['categorization'] == 'Other':
                edit_data['categorization'] = request.form.get('otherCategorization')
            save_to_json(edit_data, filename='edit.json')
            return jsonify({"status": "success", "message": "Data saved successfully"})
        elif action == 'next_page':
            # Save creativity scores
            creativity_data = {
                'page_id': page_id,
                'creativityPreEdit': request.form.get('creativityPreEdit'),
                'creativityPostEdit': request.form.get('creativityPostEdit')
            }
            save_to_json(creativity_data, filename='creativity_scores.json')
            
            # Find the next uncompleted page
            next_page_id = page_id + 1
            while next_page_id in data and data[next_page_id]['completed']:
                next_page_id += 1
            
            if next_page_id in data:
                return redirect(url_for('page', page_id=next_page_id))
            else:
                return "No more pages!", 404

    return render_template('index.html', data=data[page_id])

@app.route('/highlight_cliches/<int:page_id>', methods=['POST'])
def highlight_cliches(page_id):
    text = request.form.get('text')
    cliches = find_cliches(text)
    user_name = session.get("user_name")
    if not user_name:
        return jsonify({"status": "error", "message": "User not logged in"})

    with get_user_lock(user_name):
        filename = f'all_cliches_{user_name}.json'
        if os.path.exists(filename):
            with open(filename) as f:
                cliche_data = json.load(f)
        else:
            cliche_data = []
        cliche_data.append({'id': page_id, 'cliches': cliches})
        with open(filename, 'w') as f:
            json.dump(cliche_data, f, indent=4)
    return jsonify({"cliches": cliches})

def find_cliches(text):
    prompt = f"Given the paragraph below identify the sentences that are hackneyed and overused and unoriginal. Return a well formatted json array with keys [sentence_span,reason] and values the actual span and the reason why its cliched. There should not be any keys other than sentence_span and reason. Do not modify to sentence_span_1, sentence_span_2\n\n{text}"
    response = ''
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a creative writing assistant"}, {"role": "user", "content": prompt}])
        response = completion['choices'][0]['message']['content'].strip().replace('json', '').replace("```", '')
        print(response)
    except Exception as e:
        print("An error occurred: " + str(e))
    cliches = []
    try:
        response = json.loads(response)
        for elem in response:
            cliches.append(elem['sentence_span'])
    except Exception as e:
        print("Failed to parse json")
    return cliches

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
