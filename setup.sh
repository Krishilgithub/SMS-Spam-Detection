mkdir -p ~models/.streamlit/

echo "\
[server]\n\n
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~models/.streamlit/config.toml