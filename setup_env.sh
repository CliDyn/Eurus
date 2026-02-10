#!/bin/bash
echo "Setting up Eurus environment..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key
ARRAYLAKE_API_KEY=your_arraylake_api_key
EOL
    echo ".env file created. Please update it with your API keys."
else
    echo ".env file already exists."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -e .

echo "Setup complete!"
