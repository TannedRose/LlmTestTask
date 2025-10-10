document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');

    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const message = messageInput.value;
        if (message.trim() !== '') {
            displayMessage(message, true); // Display user message
            sendMessageToServer(message);
            messageInput.value = '';
        }
    }

    function sendMessageToServer(message) {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/send_message', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                displayMessage(response.message, false); // Display server response
            }
        };
        xhr.send(JSON.stringify({ message: message }));
    }

    function displayMessage(message, isUser) {
        const messageElement = document.createElement('div');
        messageElement.innerText = message;
        messageElement.classList.add(isUser ? 'user-message' : 'server-message');
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
