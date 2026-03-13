document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const newsText = document.getElementById('news-text');
    const resultSection = document.getElementById('result-section');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    // Result elements
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceScore = document.getElementById('confidence-score');
    const resultBar = document.getElementById('result-bar');
    const explanationText = document.getElementById('explanation-text');
    const keywordsContainer = document.getElementById('keywords-container');
    
    analyzeBtn.addEventListener('click', async () => {
        const text = newsText.value.trim();
        
        if (!text) {
            // Simple shake animation for validation
            newsText.style.animation = 'shake 0.5s cubic-bezier(.36,.07,.19,.97) both';
            setTimeout(() => newsText.style.animation = '', 500);
            return;
        }
        
        // Show loading state
        setLoading(true);
        resultSection.classList.add('hidden');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text })
            });
            
            const data = await response.json();
            
            if (response.ok && !data.error) {
                // Add an artificial small delay to show the computing animation
                setTimeout(() => {
                    displayResult(data);
                }, 600);
            } else {
                alert(data.error || 'An error occurred during analysis.');
                setLoading(false);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server. Is the backend running?');
            setLoading(false);
        }
    });
    
    clearBtn.addEventListener('click', () => {
        newsText.value = '';
        resultSection.classList.add('hidden');
        resultSection.classList.remove('is-fake', 'is-real');
    });
    
    function setLoading(isLoading) {
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            analyzeBtn.style.opacity = '0.8';
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = '1';
        }
    }
    
    function displayResult(data) {
        // Remove old classes
        resultSection.classList.remove('is-fake', 'is-real');
        
        // Set new values
        const isFake = data.prediction === 'Fake News';
        const cssClass = isFake ? 'is-fake' : 'is-real';
        
        resultSection.classList.add(cssClass);
        predictionLabel.textContent = data.prediction.toUpperCase();
        
        // Animate counter
        animateValue(confidenceScore, 0, data.confidence_score, 1200);
        
        // Animate bar (reset first)
        resultBar.style.width = '0%';
        setTimeout(() => {
            resultBar.style.width = `${data.confidence_score}%`;
        }, 50);
        
        explanationText.textContent = data.explanation;
        
        // Populate keywords
        keywordsContainer.innerHTML = '';
        if (data.key_words && data.key_words.length > 0) {
            data.key_words.forEach((word, index) => {
                const span = document.createElement('span');
                span.className = 'tag';
                span.textContent = word;
                // Add a staggered unhide animation if desired
                span.style.opacity = '0';
                span.style.transform = 'translateY(10px)';
                span.style.transition = 'all 0.3s ease';
                
                keywordsContainer.appendChild(span);
                
                setTimeout(() => {
                    span.style.opacity = '1';
                    span.style.transform = 'translateY(0)';
                }, 100 * (index + 1));
            });
        } else {
            keywordsContainer.innerHTML = '<span class="tag" style="opacity: 0.5;">No distinctive patterns</span>';
        }
        
        setLoading(false);
        resultSection.classList.remove('hidden');
        
        // Scroll to result
        setTimeout(() => {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
    
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            
            // Easing function: easeOutQuart
            const easeOutProgress = 1 - Math.pow(1 - progress, 4);
            
            obj.innerHTML = (easeOutProgress * (end - start) + start).toFixed(1);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
