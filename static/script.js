document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const predictionCard = document.querySelector('.prediction-card');

    // Form validation function
    function validateForm() {
        const inputs = form.querySelectorAll('input[type="number"], select');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!input.value || input.value.trim() === '') {
                isValid = false;
                input.classList.add('is-invalid');
            } else {
                input.classList.remove('is-invalid');
            }
        });
        
        return isValid;
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!validateForm()) {
            predictionCard.style.display = 'block';
            predictionCard.className = 'prediction-card card bg-warning text-white';
            document.getElementById('predictionText').textContent = 'Please fill in all required fields.';
            return;
        }

        const formData = {
            days_since_signup: parseInt(document.getElementById('days_since_signup').value),
            monthly_revenue: parseInt(document.getElementById('monthly_revenue').value),
            number_of_logins_last30days: parseInt(document.getElementById('number_of_logins_last30days').value),
            active_features_used: parseInt(document.getElementById('active_features_used').value),
            support_tickets_opened: parseInt(document.getElementById('support_tickets_opened').value),
            last_payment_status: document.getElementById('last_payment_status').value === '1' ? 'Failed' : 'Success',
            subscription_plan: document.getElementById('subscription_plan').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            predictionCard.style.display = 'block';
            const timeToChurn = document.getElementById('timeToChurn');
            const daysToChurn = document.getElementById('daysToChurn');
            const riskFactors = document.getElementById('riskFactors');
            const riskFactorsList = document.getElementById('riskFactorsList');
            const retentionFactors = document.getElementById('retentionFactors');
            const retentionFactorsList = document.getElementById('retentionFactorsList');

            const comprehensionHelp = document.getElementById('comprehensionHelp');
            
            if (result.status === 'error') {
                predictionCard.className = 'prediction-card card bg-danger text-white';
                document.getElementById('predictionText').textContent = result.message;
                riskFactors.style.display = 'none';
                retentionFactors.style.display = 'none';
                comprehensionHelp.style.display = 'none';
            } else {
                predictionCard.className = `prediction-card card ${result.prediction === 'Churn' ? 'bg-danger' : 'bg-success'} text-white`;
                
                // Display prediction result and comprehension score
                const comprehensionScore = result.comprehension_score || 0;
                const comprehensionEmoji = comprehensionScore >= 80 ? '🧠' : comprehensionScore >= 50 ? '🤔' : '❓';
                
                document.getElementById('predictionText').innerHTML = 
                    `${result.prediction === 'Churn' ? '⚠️ High Risk of Churn' : '✅ Low Risk of Churn'}<br>
                    <small class="mt-2 d-block">Understanding Score: ${comprehensionEmoji} ${comprehensionScore}/100</small>`;
                
                if (result.prediction === 'Churn') {
                    if (result.days_to_churn) {
                        timeToChurn.style.display = 'block';
                        daysToChurn.innerHTML = `Estimated ${result.days_to_churn} days until potential churn`;
                    }
                    
                    if (result.risk_factors && result.risk_factors.length > 0) {
                        riskFactors.style.display = 'block';
                        riskFactorsList.innerHTML = result.risk_factors
                            .map(factor => `<li>• ${factor}</li>`)
                            .join('');
                    }
                    retentionFactors.style.display = 'none';
                    
                    // Always show comprehension help section for churn predictions too
                    comprehensionHelp.style.display = 'block';
                } else {
                    if (result.retention_factors && result.retention_factors.length > 0) {
                        retentionFactors.style.display = 'block';
                        retentionFactorsList.innerHTML = result.retention_factors
                            .map(factor => `<li>• ${factor}</li>`)
                            .join('');
                    }
                    timeToChurn.style.display = 'none';
                    riskFactors.style.display = 'none';
                    
                    // Always show comprehension help section
                    comprehensionHelp.style.display = 'block';
                }
            }
        } catch (error) {
            console.error('Error:', error);
            predictionCard.style.display = 'block';
            predictionCard.className = 'prediction-card card bg-danger text-white';
            document.getElementById('predictionText').textContent = 'An error occurred while making the prediction. Please try again.';
        }
    });

    // Add input event listeners for real-time validation
    form.querySelectorAll('input[type="number"], select').forEach(input => {
        input.addEventListener('input', function() {
            if (this.value && this.value.trim() !== '') {
                this.classList.remove('is-invalid');
            }
        });
    });
});