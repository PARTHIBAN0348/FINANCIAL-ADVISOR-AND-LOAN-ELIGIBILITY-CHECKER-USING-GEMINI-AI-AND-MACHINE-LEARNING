document.getElementById('send-button').addEventListener('click', function () {
    const userInput = document.getElementById('user-input').value;
    if (!userInput) return;

    const chatBox = document.getElementById('chat-box');
    const loadingSpinner = document.getElementById('loading');

    // Clear input and show loading spinner
    document.getElementById('user-input').value = '';
    loadingSpinner.style.display = 'block';

    // Add user message to chat box
    chatBox.innerHTML += `<div class="message user-message"><strong>You:</strong> ${userInput}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message

    // Send user input to the server
    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        // Add bot response to chat box
        chatBox.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> ${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
    })
    .catch(error => {
        console.error('Error:', error);
        chatBox.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> Sorry, something went wrong. Please try again.</div>`;
    })
    .finally(() => {
        loadingSpinner.style.display = 'none'; // Hide loading spinner
    });
});