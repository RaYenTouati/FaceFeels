document.addEventListener('DOMContentLoaded', () => {
    // Elements DOM
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const imagePreview = document.getElementById('imagePreview');
    const resultContent = document.getElementById('resultContent');
    
    // Icones des émotions
    const emotionIcons = {
        happy: 'fa-face-smile',
        sadness: 'fa-face-sad-tear',
        anger: 'fa-face-angry',
        surprise: 'fa-face-surprise',
        contempt: 'fa-face-meh',
        fear: 'fa-face-fearful',
        disgust: 'fa-face-grimace'
    };
    
    // Couleurs des émotions
    const emotionColors = {
        happy: '#4CAF50',
        sadness: '#2196F3',
        anger: '#F44336',
        surprise: '#FFC107',
        contempt: '#9E9E9E',
        fear: '#673AB7',
        disgust: '#795548'
    };
    
    // Gestion de l'upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.match('image.*')) {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                imagePreview.style.display = 'block';
                analyzeBtn.disabled = false;
                resultContent.innerHTML = '<p class="placeholder">Prêt à analyse</p>';
            };
            
            reader.readAsDataURL(file);
        }
    });
    
    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });
    
    // Analyse de l'image
    analyzeBtn.addEventListener('click', async () => {
        if (!fileInput.files[0]) return;
        
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyse en cours...';
        analyzeBtn.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const response = await fetch('http://localhost:5000/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Erreur lors de l\'analyse');
            }
            
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            resultContent.innerHTML = `
                <div class="error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyser';
            analyzeBtn.disabled = false;
        }
    });
    
    // Affichage des résultats
    function displayResults(data) {
        const { emotion, probabilities } = data;
        
        // Création du HTML pour les résultats
        const resultHTML = `
            <div class="result-container result-animation">
                <div class="uploaded-image-container">
                    <img src="${imagePreview.src}" alt="Uploaded" class="uploaded-image">
                </div>
                
                <div class="emotion-result">
                    <i class="fas ${emotionIcons[emotion]} emotion-icon ${emotion}"></i>
                    <h2 class="emotion-text ${emotion}">${emotion}</h2>
                    
                    <div class="probability-container">
                        ${Object.entries(probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .map(([emo, prob]) => `
                                <div class="probability-item">
                                    <div class="probability-label">
                                        <span>${emo}</span>
                                        <span>${(prob * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="probability-bar">
                                        <div class="probability-fill ${emo}" 
                                             style="width: ${prob * 100}%; 
                                             background: ${emotionColors[emo]}">
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                    </div>
                    
                    <button class="music-button" onclick="playMusic('${emotion}')">
                        <i class="fas fa-music"></i>
                        Play ${emotion} Music
                    </button>
                </div>
            </div>
        `;
        
        resultContent.innerHTML = resultHTML;
    }
    
    // Jouer la musique
    async function playMusic(emotion) {
        try {
            const response = await fetch('http://localhost:5000/api/play', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ emotion })
            });
            
            if (!response.ok) {
                throw new Error('Erreur lors de la lecture');
            }
            
            const data = await response.json();
            console.log(data.message);
        } catch (error) {
            console.error('Erreur musique:', error);
        }
    }
});