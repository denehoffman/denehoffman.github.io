default:
    @just --list

# Start the local development server and open in a browser.
open:
    zola serve --open

# Start the local development server.
serve:
    zola serve

# Build the minified production site.
build:
    zola build

# Serve a production-style build without opening a browser.
preview-production:
    zola serve --interface 127.0.0.1 --port 1111

# Run deterministic local checks (external URLs are deliberately separate).
check:
    python3 scripts/validate_site.py
    zola build
    npm test

# Ask Zola to verify external links; network failures can make this noisy.
check-links:
    zola check

# Run the browser smoke and accessibility suite.
test-browser:
    npm test

# Create content/blog/YYYY-MM-DD-SLUG/index.md.
new-post slug title:
    python3 scripts/new_content.py post "{{ slug }}" --title "{{ title }}"

# Create a project page and add it to data/projects.json.
new-project slug title summary:
    python3 scripts/new_content.py project "{{ slug }}" --title "{{ title }}" --summary "{{ summary }}"

# Refresh the checked-in Google Scholar snapshot.
update-scholar:
    python3 scripts/update_scholar.py
