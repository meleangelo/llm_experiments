# if you don't have openai package installed
#install.packages("openai")
library(openai)


# to run this code you need an OpenAI key for the API.
Sys.setenv(
  OPENAI_API_KEY = 'YOUR-API-KEY'
)

# job posting is here;
# https://www.linkedin.com/jobs/search/?alertAction=viewjobs&currentJobId=4184890670&distance=25&f_TPR=a1743783721-&geoId=103644278&keywords=social%20network%20analysis&origin=JOB_ALERT_IN_APP_NOTIFICATION&originToLandingJobPostings=4040214027&savedSearchId=7456866378&sortBy=R

# information from the job postings (you could extract it as well)
job_title = "Associate Manager, People Scientist"
company = "The Clorox Company"
job_description = "Clorox is the place that’s committed to growth – for our people and our brands. Guided by our purpose and values, and with people at the center of everything we do, we believe every one of us can make a positive impact on consumers, communities, and teammates. Join our team. #CloroxIsThePlace

Your role at Clorox:

Your role at Clorox:

The People Analytics and Insights team at Clorox collaborates with leaders across the company and the People& organization to provide data-driven insights that inform decisions related to workforce, talent, and organizational strategy. This role involves working closely with various cross-functional stakeholders to effectively distill ad hoc requests, ask pertinent questions to understand hypotheses, and respond with logical, organized data and insights in a timely manner.

We are in the early phases of establishing this function, and you will play a key role in pioneering this initiative. The ideal candidate will possess a natural curiosity and strong critical thinking and analytical skills. They will excel at managing multiple projects simultaneously while working with a global team to achieve impactful outcomes. Additionally, they will enjoy using data and insights to inform and influence stakeholders. This position reports to the Senior Manager, People Analytics and Insights.

In this role, you will:

In this role, you will:

Build trust with key stakeholders across People& and Clorox by through a consultative approach. 
Collaborate with leadership to plan and drive critical strategic, organizational, and operational initiatives for key programs. 
Apply qualitative and quantitative research methodologies to generate actionable insights that improve employee experience, engagement, and business outcomes. 
Develop and implement robust measurement strategies to evaluate the effectiveness of People& programs and initiatives. 
Present findings and recommendations to stakeholders through clear and detailed reports and presentations. 
Ensure data integrity and make informed analytical decisions. 
"
qualifications = "What we look for:

Bachelor’s degree in a quantitative field (statistics, industrial/organizational psychology, economics, engineering, etc.) with 7+ years of relevant experience 
Experience in advanced statistical methods (e.g., regression, cluster analysis, HLM, social network analysis, text analysis, longitudinal methods, etc.). 
Experience working with people data (e.g., survey data, HR data, psychological study data, human behavior data) in an applied setting. 
Proven ability to translate data into compelling and easy-to-understand insights, stories, and narratives for a non-technical audience. 
Proficiency in Python or R and SQL queries. 
Excellent execution and organizational skills and attention to detail. 

Additional Qualifications that are nice to have: 

Masters in a quantitative field (statistics, industrial/organizational psychology, economics, engineering, etc.) 
"

# now we run our prompt to create a personalized cover letter
answer = create_chat_completion(
  model = "gpt-4o",    # model to use
  temperature = 0,   # temp
  messages = list(
    list(
      "role" = "system",
      #"content" = "You are an AI assistant helping me answering questions."
      "content" = paste("You are an AI assistant helping to write a 
                        cover letter for a job application.",
                        "The job ad details are as follows:",
                        paste("Job Title:", job_title),
                        paste("Company:", company),
                        paste("Job Description:", job_description),
                        paste("Required Qualifications:", qualifications), sep = "\n")
    ),
    list(
      "role" = "user",
      "content" = "Using these data, write a cover letter for 
      the job in the style of Donald Trump: bold, confident, 
      using superlatives like 'tremendous', 'fantastic', 
      'the best', and 'yuge', with a brash, self-promoting tone. 
      Exaggerate achievements and qualifications and make it sound like 
      a winning pitch."
    )
  )
)


cat(answer$choices$message.content)

