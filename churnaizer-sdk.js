window.Churnaizer = {
  track: async function(userData, apiKey, callback) {
    async function retryFetch(url, options, retries = 2) {
      for (let i = 0; i <= retries; i++) {
        try {
          const response = await fetch(url, options);
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
          }
          const contentType = response.headers.get("content-type");
          if (!contentType || !contentType.includes("application/json")) {
            const responseText = await response.text();
            throw new Error(`Expected JSON response, but received ${contentType || 'no content type'}. Response: ${responseText.substring(0, 200)}...`);
          }
          return await response.json();
        } catch (e) {
          if (i === retries) throw e;
        }
      }
    }

    if (!userData || typeof userData !== "object" || !userData.user_id || typeof userData.user_id !== 'string' || userData.user_id.trim() === '') {
      console.error("\n❌\nChurnaizer SDK: userData must include a valid user_id.");
      if (callback) callback(null, new Error("userData must include a valid user_id."));
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000); // 5 seconds timeout

    try {
      // Step 1: Send data to AI model to get churn prediction
      const data = await retryFetch("https://ai-model-rumc.onrender.com/api/v1/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": apiKey,
          "X-SDK-Version": "1.0.0"
        },
        body: JSON.stringify(userData),
        signal: controller.signal
      });

      clearTimeout(timeout);

      // Validate the structure of the AI model's response
      const requiredFields = ['churn_probability', 'reason', 'message', 'understanding_score'];
      for (const field of requiredFields) {
        if (!(field in data)) {
          throw new Error(`AI model response missing required field: ${field}. Full response: ${JSON.stringify(data)}`);
        }
      }

      const churn_score = data.churn_probability;
      const churn_reason = data.reason;
      const insight = data.message;
      const understanding = data.understanding_score;

      // Step 2: Sync this data with Churnaizer backend
       fetch("https://churnaizer.com/api/sync", {
         method: "POST",
         headers: {
           "Content-Type": "application/json"
         },
         body: JSON.stringify({
           ...userData,
           churn_score,
           churn_reason,
           insight,
           understanding
         })
       }).catch(syncError => {
         console.error("❌ Churnaizer SDK: Sync to /api/sync failed:", syncError);
       });


      // Step 3: Callback to show in browser
      if (callback) {
        callback({ churn_score, churn_reason, insight, understanding });
      }
    } catch (error) {
      clearTimeout(timeout);
      if (callback) callback(null, error);
      console.error("\n❌\nChurnaizer SDK tracking failed:", error);
    }
  }
};