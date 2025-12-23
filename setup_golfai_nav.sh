#!/bin/bash

# Detect shell
SHELL_RC=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    echo "Unsupported shell. Only Bash and Zsh supported."
    exit 1
fi

echo "Using shell config: $SHELL_RC"

# Backup existing config
cp "$SHELL_RC" "${SHELL_RC}.backup_$(date +%s)"

# Add GolfAIProject navigation function
cat <<'EOL' >> "$SHELL_RC"

# GolfAIProject folder jump with tab completion
g() {
    case "$1" in
        ""|home) cd ~/GolfAIProject ;;
        data) cd ~/GolfAIProject/data ;;
        videos|v) cd ~/GolfAIProject/data/videos ;;
        a|a6700) cd ~/GolfAIProject/data/videos/a6700 ;;
        i|instacam) cd ~/GolfAIProject/data/videos/instacam ;;
        p1|phone1) cd ~/GolfAIProject/data/videos/phone1 ;;
        p2|phone2) cd ~/GolfAIProject/data/videos/phone2 ;;
        n|annotations) cd ~/GolfAIProject/data/annotations ;;
        m|models) cd ~/GolfAIProject/data/models ;;
        s|scripts) cd ~/GolfAIProject/data/scripts ;;
        *) echo "Unknown folder: $1" ;;
    esac
}

# Bash tab completion
if [ -n "$BASH_VERSION" ]; then
    _g_completions() {
        local cur="${COMP_WORDS[COMP_CWORD]}"
        local options="home data videos v a a6700 i instacam p1 p2 n m s"
        COMPREPLY=( $(compgen -W "$options" -- "$cur") )
    }
    complete -F _g_completions g
fi

# Zsh tab completion
if [ -n "$ZSH_VERSION" ]; then
    _g_completions() {
        reply=( home data videos v a a6700 i instacam p1 p2 n m s )
    }
    compctl -K _g_completions g
fi

EOL

# Reload shell config
source "$SHELL_RC"

echo "GolfAIProject navigation setup complete! Use 'g' to jump folders."
