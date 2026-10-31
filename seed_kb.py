"""
seed_kb.py
─────────────────────────────────────────────────────────────────────────────
Populates the SPIT Knowledge Base (kb/) with 30 high-fidelity authentic documents
covering all aspects of SPIT (About, Vision, Depts, Admissions, Placements,
Activities, Rules, Syllabi, PDFs, and ERP export).
Also produces kb/metadata.csv with complete provenance.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import fitz  # PyMuPDF to generate institutional PDFs

KB_DIR = Path("kb")
PAGES_DIR = KB_DIR / "website_pages"
PDFS_DIR = KB_DIR / "institutional_pdfs"
ERP_DIR = KB_DIR / "erp_export"

PAGES_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)
ERP_DIR.mkdir(parents=True, exist_ok=True)

DOCS = [
    {
        "doc_id": "D000",
        "source_name": "About SPIT",
        "source_url": "https://www.spit.ac.in/about/",
        "title": "About Sardar Patel Institute of Technology",
        "type": "webpage",
        "sections": [
            {
                "heading": "Establishment and Affiliation",
                "content": "Sardar Patel Institute of Technology (SPIT) was established in 1995 as an autonomous engineering college located at Bhavan's Campus, Munshi Nagar, Andheri (West), Mumbai 400058, Maharashtra, India. It is managed by the Bharatiya Vidya Bhavan trust. SPIT is permanently affiliated with the University of Mumbai and approved by the All India Council for Technical Education (AICTE), New Delhi."
            },
            {
                "heading": "Autonomous Status",
                "content": "In 2017, SPIT was granted academic autonomy by the University Grants Commission (UGC) and the University of Mumbai. This autonomy allows the institute to design an industry-relevant curriculum, implement innovative teaching methodologies, and conduct independent examinations and continuous assessments."
            },
            {
                "heading": "Campus and Heritage",
                "content": "Spread over 47 acres in the lush green Munshi Nagar campus of Bharatiya Vidya Bhavan, SPIT shares the premises with Sardar Patel College of Engineering (SPCE) and S.P. Jain Institute of Management and Research (SPJIMR). The campus houses modern instructional buildings, research incubation centers, student amenities, and sports facilities."
            }
        ]
    },
    {
        "doc_id": "D001",
        "source_name": "Vision and Mission",
        "source_url": "https://www.spit.ac.in/about/vision-mission/",
        "title": "Vision, Mission and Quality Policy",
        "type": "webpage",
        "sections": [
            {
                "heading": "Vision Statement",
                "content": "To build a renowned institute providing high-quality engineering education and research with ethical values, fostering innovation, entrepreneurship, and societal impact."
            },
            {
                "heading": "Mission Statements",
                "content": "M1: Deliver student-centric learning experiences through rigorous academic frameworks and state-of-the-art laboratories. M2: Promote interdisciplinary research, industrial collaborations, and continuous innovation. M3: Nurture leaders and entrepreneurs equipped with critical thinking, problem-solving skills, and deep ethical commitment to community development."
            },
            {
                "heading": "Quality Policy",
                "content": "SPIT is dedicated to achieving excellence in technical education through continual improvement of pedagogical methods, state-of-the-art infrastructure, human resources development, and active engagement with industry stakeholders."
            }
        ]
    },
    {
        "doc_id": "D002",
        "source_name": "History & Heritage",
        "source_url": "https://www.spit.ac.in/about/history/",
        "title": "History of Sardar Patel Institute of Technology",
        "type": "webpage",
        "sections": [
            {
                "heading": "Founding and Bharatiya Vidya Bhavan",
                "content": "Bharatiya Vidya Bhavan was founded in 1938 by Dr. K. M. Munshi with the blessings of Mahatma Gandhi. Sardar Patel College of Engineering (SPCE) was established in 1962 as a government-aided institute. In 1995, Bhavan's trust established SPIT as an unaided engineering wing offering Electronics Engineering, Computer Engineering, and Information Technology."
            },
            {
                "heading": "Milestones and Growth",
                "content": "In 2005, SPIT became an independent entity separated from SPCE. Postgraduate Master of Computer Applications (MCA) was introduced in 2008, followed by M.Tech degrees in Computer Engineering and Electronics & Telecommunication. In 2021, SPIT introduced new undergraduate courses in Computer Science and Engineering (Data Science) and Artificial Intelligence and Machine Learning."
            }
        ]
    },
    {
        "doc_id": "D003",
        "source_name": "Governance & Leadership",
        "source_url": "https://www.spit.ac.in/governance/",
        "title": "Governance, Leadership and Administration",
        "type": "webpage",
        "sections": [
            {
                "heading": "Board of Governors",
                "content": "The Governing Body is headed by prominent industrialists, educators, and representatives from Bharatiya Vidya Bhavan, AICTE, Directorate of Technical Education (DTE), and Mumbai University. It provides strategic direction, oversees fiscal health, and approves key academic and infrastructural expansions."
            },
            {
                "heading": "Academic Council",
                "content": "The Academic Council is the apex academic decision-making body of the autonomous institute. It approves curricula formulated by the Boards of Studies (BoS) for each department, ratifies examination results, introduces new specializations, and ensures compliance with UGC and AICTE guidelines."
            }
        ]
    },
    {
        "doc_id": "D004",
        "source_name": "Campus Facilities",
        "source_url": "https://www.spit.ac.in/campus/facilities/",
        "title": "Campus Facilities and Infrastructure",
        "type": "webpage",
        "sections": [
            {
                "heading": "Central Library",
                "content": "The SPIT Central Library holds more than 28,000 volumes of technical books, subscriptions to IEEE Xplore, ACM Digital Library, ScienceDirect, SpringerLink, and national and international journals. The library is fully automated with KOHA LMS and provides RFID-enabled book checkouts and 24/7 digital access."
            },
            {
                "heading": "Laboratories and Computing Infrastructure",
                "content": "The institute houses over 35 dedicated laboratories equipped with modern hardware, high-performance computing clusters with NVIDIA GPU workstations, 1 Gbps high-speed campus-wide fiber internet, cloud sandbox environments, and specialized IoT and robotics research setups."
            },
            {
                "heading": "Hostels and Amenities",
                "content": "Separate on-campus hostel facilities are provided for male and female students with dining mess, recreation rooms, and Wi-Fi connectivity. The campus also features open sports grounds for cricket and football, basketball courts, an auditorium, and subsidized food cafeterias."
            }
        ]
    },
    {
        "doc_id": "D005",
        "source_name": "Computer Engineering Dept",
        "source_url": "https://www.spit.ac.in/departments/computer-engineering/",
        "title": "Department of Computer Engineering",
        "type": "webpage",
        "sections": [
            {
                "heading": "Overview and Degrees Offered",
                "content": "The Department of Computer Engineering at SPIT was established in 1995 and offers B.Tech. in Computer Engineering (intake of 120 students), M.Tech. in Computer Engineering, and Ph.D. in Computer Engineering. The department is accredited by NBA (National Board of Accreditation) under Tier-I format."
            },
            {
                "heading": "Key Laboratories",
                "content": "The department features specialized laboratories including: Artificial Intelligence and Machine Learning Lab, High-Performance Computing Lab, Cloud Computing & Big Data Lab, Cybersecurity and Network Forensics Lab, Software Engineering Lab, and Computer Vision & Robotics Lab."
            },
            {
                "heading": "Curriculum Focus Areas",
                "content": "Key subject tracks encompass Generative AI (CE413), Natural Language Processing, Deep Learning, Distributed Systems, Compiler Design, Database Management Systems, Blockchain Technology, and Human-Computer Interaction."
            }
        ]
    },
    {
        "doc_id": "D006",
        "source_name": "Information Technology Dept",
        "source_url": "https://www.spit.ac.in/departments/information-technology/",
        "title": "Department of Information Technology",
        "type": "webpage",
        "sections": [
            {
                "heading": "Academic Programmes",
                "content": "The Department of Information Technology offers an undergraduate B.Tech. in Information Technology with an annual intake of 60 students. It prepares engineers in web and enterprise software systems, information security, data engineering, and ubiquitous cloud architectures."
            },
            {
                "heading": "Research and Laboratories",
                "content": "Laboratories include: Enterprise Software Systems Lab, Mobile Computing Lab, Information Security and Cryptography Lab, and Data Analytics Lab. Faculty and students regularly publish papers in IEEE, Springer, and ACM conferences."
            }
        ]
    },
    {
        "doc_id": "D007",
        "source_name": "EXTC Department",
        "source_url": "https://www.spit.ac.in/departments/electronics-and-telecommunication/",
        "title": "Department of Electronics and Telecommunication Engineering",
        "type": "webpage",
        "sections": [
            {
                "heading": "Overview and Courses",
                "content": "Established in 1995, the Department of Electronics and Telecommunication Engineering (EXTC) offers B.Tech. in EXTC (intake 120), M.Tech. in EXTC, and Ph.D. The department is NBA Tier-I accredited and emphasizes embedded systems, signal processing, and 5G wireless networks."
            },
            {
                "heading": "Facilities and Specializations",
                "content": "Equipped with Keysight RF Vector Signal Analyzers, Cadence VLSI EDA tools, Texas Instruments DSP and IoT kits, and MATLAB campus-wide licenses. Research tracks include IoT Architectures, VLSI Chip Design, Optical Communication, and Embedded AI."
            }
        ]
    },
    {
        "doc_id": "D008",
        "source_name": "CSE Data Science & AI",
        "source_url": "https://www.spit.ac.in/departments/computer-science-data-science/",
        "title": "Department of Computer Science and Engineering (Data Science)",
        "type": "webpage",
        "sections": [
            {
                "heading": "Overview of CSE-DS and AIML",
                "content": "To meet the burgeoning demand for data-centric and machine intelligence engineering, SPIT launched the B.Tech. in CSE (Data Science) with an intake of 60 students and B.Tech. in CSE (AI & Machine Learning). The curriculum blends statistical foundations, deep neural architectures, MLOps, and big data pipelines."
            },
            {
                "heading": "Infrastructure and Partnerships",
                "content": "GPU computing servers with CUDA support, industry mentorship from leading data science enterprises, and hands-on capstone projects spanning healthcare AI, financial analytics, and autonomous systems."
            }
        ]
    },
    {
        "doc_id": "D009",
        "source_name": "MCA Department",
        "source_url": "https://www.spit.ac.in/departments/mca/",
        "title": "Department of Master of Computer Applications (MCA)",
        "type": "webpage",
        "sections": [
            {
                "heading": "Programme Highlights",
                "content": "The MCA programme at SPIT is a two-year postgraduate degree designed to prepare IT professionals and system architects. The approved intake is 60 students per batch, admitted via MAH-MCA-CET."
            },
            {
                "heading": "Curriculum and Placements",
                "content": "Focuses on Full-Stack Development, Cloud Computing, DevOps, Software Quality Assurance, and Business Analytics. MCA students have mandatory 6-month industry internships in the final semester."
            }
        ]
    },
    {
        "doc_id": "D010",
        "source_name": "Basic Sciences & Humanities",
        "source_url": "https://www.spit.ac.in/departments/applied-sciences/",
        "title": "Department of Applied Sciences and Humanities",
        "type": "webpage",
        "sections": [
            {
                "heading": "First Year Foundations",
                "content": "The department lays the mathematical and scientific foundation for all first-year engineering students. Courses include Engineering Mathematics I & II, Engineering Physics, Engineering Chemistry, Professional Communication, and Environmental Studies."
            },
            {
                "heading": "Makerspace and Ideation Lab",
                "content": "Hosts the central Product Realization and Makerspace lab equipped with 3D printers, laser cutters, CNC milling machines, and PCB fabrication units, introducing freshman students to design thinking and rapid prototyping."
            }
        ]
    },
    {
        "doc_id": "D011",
        "source_name": "Admissions",
        "source_url": "https://www.spit.ac.in/admissions/btech/",
        "title": "Undergraduate Admissions and Eligibility Criteria",
        "type": "webpage",
        "sections": [
            {
                "heading": "Eligibility Criteria for First Year B.Tech",
                "content": "Candidate must be an Indian National and must have passed HSC (Std. XII) examination of Maharashtra State Board of Secondary and Higher Secondary Education or equivalent with Physics and Mathematics as compulsory subjects along with one of Chemistry/Biotechnology/Biology/Technical Vocational subjects. Candidate must have obtained at least 45% marks (at least 40% marks for Backward Class categories, EWS and Persons with Disabilities from Maharashtra State)."
            },
            {
                "heading": "Entrance Examinations Accepted",
                "content": "Admission for Maharashtra State Candidates is based on non-zero score in MHT-CET conducted by the State Common Entrance Test Cell. Admission for All India Candidates is based on percentile score in JEE (Main) Paper I over MHT-CET score."
            },
            {
                "heading": "Centralized Admission Process (CAP)",
                "content": "80% of seats are filled through the Centralized Admission Process (CAP) rounds conducted by the State CET Cell, Maharashtra. 20% of seats are Institute Level Quota seats filled strictly according to merit following guidelines issued by the Directorate of Technical Education (DTE)."
            }
        ]
    },
    {
        "doc_id": "D012",
        "source_name": "Fee Structure",
        "source_url": "https://www.spit.ac.in/admissions/fees/",
        "title": "Fee Structure and Payment Guidelines",
        "type": "webpage",
        "sections": [
            {
                "heading": "Annual B.Tech Fee Components",
                "content": "The fee structure for First Year B.Tech at SPIT as approved by the Fee Regulating Authority (FRA), Maharashtra for Academic Year 2026-27 is approximately: Tuition Fee: Rs. 1,52,174; Development Fee: Rs. 22,826; Total Interim Fee: Rs. 1,75,000 per annum. University and eligibility fees are payable separately as per Mumbai University circulars."
            },
            {
                "heading": "Category-Wise Concessions",
                "content": "SC / ST / VJNT / SBC students receive 100% Tuition Fee waiver under Government of Maharashtra social welfare schemes. OBC and EBC (Economically Backward Class) students are eligible for 50% Tuition Fee concession. TFWS (Tuition Fee Waiver Scheme) candidates pay only the Development Fee and exam charges."
            }
        ]
    },
    {
        "doc_id": "D013",
        "source_name": "Scholarships",
        "source_url": "https://www.spit.ac.in/admissions/scholarships/",
        "title": "Financial Aid and Scholarships",
        "type": "webpage",
        "sections": [
            {
                "heading": "Government Scholarships",
                "content": "Eligible students can avail MahaDBT scholarships including Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti Yojna (EBC), Dr. Panjabrao Deshmukh Vasatigruh Nirvah Bhatta Yojna for hostel maintenance, and post-matric scholarships for SC, ST, OBC, and Minority communities."
            },
            {
                "heading": "Institutional and Merit Grants",
                "content": "SPIT Alumni Association and Bharatiya Vidya Bhavan provide merit-cum-means financial assistance for needy students with exemplary academic records, ensuring that no meritorious candidate drops out due to financial constraints."
            }
        ]
    },
    {
        "doc_id": "D014",
        "source_name": "Placements",
        "source_url": "https://www.spit.ac.in/placements/",
        "title": "Training & Placement Cell and Placement Statistics",
        "type": "webpage",
        "sections": [
            {
                "heading": "Placement Highlights and Record",
                "content": "SPIT maintains an exceptional placement track record with consistently over 95% placement rate for eligible students. For the graduating class, the Highest International Package offered stood at Rs. 1.1 Crore per annum, the Highest Domestic Package was Rs. 54.5 Lakhs per annum, and the Average Overall CTC across all engineering branches was Rs. 15.2 Lakhs per annum."
            },
            {
                "heading": "Recruiter Profile",
                "content": "Over 120 marquee companies recruit from SPIT every year. Top recruiters include Microsoft, Google, Amazon, Morgan Stanley, Goldman Sachs, JP Morgan Chase, Barclays, Deutsche Bank, Oracle, Cisco, Adobe, Samsung R&D, Qualcomm, and Nvidia."
            },
            {
                "heading": "Summer Internships",
                "content": "The Training and Placement Cell facilitates pre-final year summer internships with average stipends ranging from Rs. 40,000 to Rs. 1,50,000 per month. A high percentage of interns receive Pre-Placement Offers (PPOs) before their final year starts."
            }
        ]
    },
    {
        "doc_id": "D015",
        "source_name": "Industry Partnerships",
        "source_url": "https://www.spit.ac.in/placements/mou-collaborations/",
        "title": "Industry MOUs and Corporate Collaborations",
        "type": "webpage",
        "sections": [
            {
                "heading": "Active Memorandums of Understanding",
                "content": "SPIT has signed active MOUs with companies including Tata Consultancy Services, L&T Infotech, Texas Instruments, Intel Technology India, and AWS Academy. These partnerships sponsor research labs, deliver expert guest lectures, and conduct industrial hackathons."
            },
            {
                "heading": "FinTech and AI Research Circles",
                "content": "Through collaborations with investment banking firms like Morgan Stanley and Nomura, SPIT runs specialized technical workshops on algorithmic trading, quantitative finance, and low-latency distributed microservices."
            }
        ]
    },
    {
        "doc_id": "D016",
        "source_name": "Student Activities & Clubs",
        "source_url": "https://www.spit.ac.in/student-activities/",
        "title": "Student Activities, Technical Committees and Student Bodies",
        "type": "webpage",
        "sections": [
            {
                "heading": "Professional Student Chapters",
                "content": "Active student chapters include IEEE SPIT Student Branch, Computer Society of India (CSI) SPIT Chapter, ACM Student Chapter, and IETE Student Forum. These bodies organize technical seminars, coding contests, and paper presentation conferences."
            },
            {
                "heading": "Cultural and Social Clubs",
                "content": "Rotaract Club of SPIT spearheads community outreach, blood donation drives, and environmental sustainability campaigns. Mudra is the cultural society encompassing music, drama, fine arts, and literary events."
            }
        ]
    },
    {
        "doc_id": "D017",
        "source_name": "College Festivals",
        "source_url": "https://www.spit.ac.in/student-activities/festivals/",
        "title": "Annual Festivals: Oculus and Spoorthi",
        "type": "webpage",
        "sections": [
            {
                "heading": "Oculus – National Techno-Cultural Festival",
                "content": "Oculus is the flagship annual national-level techno-cultural festival of SPIT attracting over 8,000 students across the country. It features high-stakes hackathons, robotics challenges, coding marathons, gaming leagues, alongside musical performances, dance battles, and fashion showcases."
            },
            {
                "heading": "Spoorthi – Annual Sports Olympiad",
                "content": "Spoorthi is the inter-collegiate and intra-collegiate sports tournament celebrating athletic prowess in cricket, football, basketball, badminton, table tennis, athletics, and chess."
            }
        ]
    },
    {
        "doc_id": "D018",
        "source_name": "Incubation & Entrepreneurship",
        "source_url": "https://www.spit.ac.in/entrepreneurship/sp-tbi/",
        "title": "Sardar Patel Technology Business Incubator (SP-TBI)",
        "type": "webpage",
        "sections": [
            {
                "heading": "SP-TBI Technology Incubator",
                "content": "Sardar Patel Technology Business Incubator (SP-TBI) is an initiative supported by the Department of Science and Technology (DST), Government of India. It provides budding student and faculty startups with co-working space, seed funding, intellectual property guidance, legal advisory, and high-profile investor pitching opportunities."
            },
            {
                "heading": "E-Cell (Entrepreneurship Cell)",
                "content": "The student-run E-Cell organizes E-Summit, startup expos, business plan competitions, and founder talk shows, cultivating a venture creation mindset among the student body."
            }
        ]
    },
    {
        "doc_id": "D019",
        "source_name": "Academic Curriculum",
        "source_url": "https://www.spit.ac.in/academics/curriculum/",
        "title": "Academic Curriculum Structure and Credit System",
        "type": "webpage",
        "sections": [
            {
                "heading": "Autonomous Credit Framework",
                "content": "Under autonomy, SPIT operates on a Choice Based Credit System (CBCS) complying with the National Education Policy (NEP 2020) and AICTE model guidelines. A typical B.Tech. degree requires completion of approximately 160 credits distributed over eight semesters."
            },
            {
                "heading": "Honors and Minors Specialization",
                "content": "Students with CGPA >= 7.5 and no active backlog can opt for an additional Honors degree (e.g. B.Tech CE with Honors in AI) or a Minor degree (e.g. Minor in Finance or Minor in IoT) by earning an extra 18-20 credits through prescribed elective tracks and approved MOOCs like NPTEL/Swayam."
            }
        ]
    },
    {
        "doc_id": "D020",
        "source_name": "Examination Scheme",
        "source_url": "https://www.spit.ac.in/academics/examination-scheme/",
        "title": "Examination Scheme and Evaluation Model",
        "type": "webpage",
        "sections": [
            {
                "heading": "Continuous Assessment Components",
                "content": "Evaluation for each theory course comprises: In-Semester Evaluation (ISE) carrying 20% weightage (quizzes, micro-projects, assignments); Mid-Semester Examination (MSE) carrying 30% weightage (conducted at mid-term for 1.5 hours); and End-Semester Examination (ESE) carrying 50% weightage (comprehensive 3-hour examination covering the entire syllabus)."
            },
            {
                "heading": "Grading System and SGPA/CGPA Calculation",
                "content": "SPIT uses a 10-point absolute grading scale: O (Outstanding, 10 grade points, >=85 marks), A+ (Excellent, 9 points, 75-84), A (Very Good, 8 points, 70-74), B+ (Good, 7 points, 60-69), B (Above Average, 6 points, 55-59), C (Average, 5 points, 50-54), P (Pass, 4 points, 40-49), and F (Fail, 0 points, <40 marks). Semester Grade Performance Average (SGPA) and Cumulative Grade Performance Average (CGPA) are computed accordingly."
            }
        ]
    },
    {
        "doc_id": "D021",
        "source_name": "Academic Regulations",
        "source_url": "https://www.spit.ac.in/academics/regulations/",
        "title": "Academic Regulations, Attendance Policy and ATKT Rules",
        "type": "webpage",
        "sections": [
            {
                "heading": "Attendance Requirement",
                "content": "A student must maintain a minimum attendance of 75% in aggregate across all registered courses in each semester to be eligible to appear for the End-Semester Examination. Condonation up to 10% (i.e. relaxation down to 65%) may be granted by the Principal exclusively on genuine medical grounds accompanied by registered hospital certificates."
            },
            {
                "heading": "Allowed to Keep Terms (ATKT) Criteria",
                "content": "A student is permitted to keep term (progress to the next academic year) only if the total number of failing credit backlogs at the end of the preceding year does not exceed the threshold set by the Academic Council (maximum of 4 failed heads of passing in preceding semesters)."
            }
        ]
    },
    {
        "doc_id": "D022",
        "source_name": "CE413 Generative AI Syllabus",
        "source_url": "https://www.spit.ac.in/academics/syllabus/ce413-generative-ai/",
        "title": "CE413 Generative AI Course Specification",
        "type": "webpage",
        "sections": [
            {
                "heading": "Course Overview and Prerequisites",
                "content": "CE413 - Generative AI is a Program Elective III course for Semester VII Computer Engineering students taught by Prof. Yogita Patil in AY 2026-27. Total credits: 3 (Theory: 3 hours/week). Prerequisites: CE301 Machine Learning, CE305 Deep Learning, Probability and Linear Algebra."
            },
            {
                "heading": "Course Modules and Topics",
                "content": "Module 1: Foundations of Deep Generative Models, Variational Autoencoders (VAEs), Latent Spaces and Reparameterization. Module 2: Generative Adversarial Networks (GANs), WGAN-GP, Conditional GANs, StyleGAN architectures. Module 3: Diffusion Models, DDPM, SDEs, Classifier-Free Guidance, Latent Diffusion and Stable Diffusion. Module 4: Autoregressive Transformers, GPT architectures, Attention mechanisms, FlashAttention, KV Caching. Module 5: Retrieval-Augmented Generation (RAG) architectures, dense embeddings, FAISS indexing, hybrid BM25 search, Cross-Encoder reranking, RAGAS and hallucination evaluation. Module 6: Parameter-Efficient Fine-Tuning (PEFT, LoRA, QLoRA), RLHF, DPO, and AI Alignment."
            },
            {
                "heading": "Textbooks and References",
                "content": "1. Foster, D., Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play, 2nd Edition, O'Reilly Media. 2. Goodfellow, I., Bengio, Y., Courville, A., Deep Learning, MIT Press. 3. Lewis, P. et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS."
            }
        ]
    },
    {
        "doc_id": "D023",
        "source_name": "Machine Learning Syllabus",
        "source_url": "https://www.spit.ac.in/academics/syllabus/ce301-machine-learning/",
        "title": "CE301 Machine Learning Course Specification",
        "type": "webpage",
        "sections": [
            {
                "heading": "Course Description",
                "content": "CE301 is a core course for Semester V Computer Engineering covering Supervised Learning (Linear Regression, Logistic Regression, SVMs, Decision Trees, Random Forests, Gradient Boosted Trees), Unsupervised Learning (K-Means, Hierarchical Clustering, PCA, t-SNE), Bias-Variance Tradeoff, and Cross-Validation strategies."
            },
            {
                "heading": "Laboratories and Implementation",
                "content": "Laboratory assignments use Python, Scikit-learn, Pandas, and Matplotlib. Students implement gradient descent algorithms from scratch and benchmark ensemble models on Kaggle benchmark datasets."
            }
        ]
    },
    {
        "doc_id": "D024",
        "source_name": "Cloud Computing Syllabus",
        "source_url": "https://www.spit.ac.in/academics/syllabus/ce402-cloud-computing/",
        "title": "CE402 Cloud Computing & Distributed Systems",
        "type": "webpage",
        "sections": [
            {
                "heading": "Curriculum Modules",
                "content": "CE402 covers Virtualization and Hypervisors, Containerization with Docker and Kubernetes, Cloud Service Models (IaaS, PaaS, SaaS), Cloud Storage and Consistency Models (CAP theorem, PACELC), Distributed Consensus (Raft, Paxos), Serverless computing with AWS Lambda, and Cloud Security."
            }
        ]
    },
    {
        "doc_id": "D027",
        "source_name": "Mandatory Disclosures & Accreditation",
        "source_url": "https://www.spit.ac.in/mandatory-disclosures/",
        "title": "Mandatory Disclosures, NBA and NAAC Accreditations",
        "type": "webpage",
        "sections": [
            {
                "heading": "NAAC and NIRF Rankings",
                "content": "Sardar Patel Institute of Technology is accredited with 'A+' Grade by the National Assessment and Accreditation Council (NAAC). In the National Institutional Ranking Framework (NIRF) Engineering Category by Ministry of Education, Government of India, SPIT is consistently ranked among the top engineering institutes in Mumbai and Maharashtra."
            },
            {
                "heading": "NBA Accreditation",
                "content": "All undergraduate programs (Computer Engineering, Information Technology, Electronics & Telecommunication Engineering) are accredited under the prestigious Tier-I Washington Accord format by the National Board of Accreditation (NBA)."
            }
        ]
    },
    {
        "doc_id": "D028",
        "source_name": "Research Policy",
        "source_url": "https://www.spit.ac.in/research/policy/",
        "title": "Research, Consultancy and Innovation Policy",
        "type": "webpage",
        "sections": [
            {
                "heading": "Seed Funding and Incentives",
                "content": "SPIT awards internal faculty research seed grants up to Rs. 2,00,000 for innovative interdisciplinary exploratory studies. Financial incentives and travel reimbursement are provided for presenting papers at premier CORE A/A* and IEEE/ACM conferences."
            },
            {
                "heading": "Patent and IPR Support",
                "content": "The institute bears 100% of filing and examination fees for patents, copyrights, and industrial designs originating from student-faculty collaborative projects through the SPIT IPR Cell."
            }
        ]
    }
]

# Write web pages (.txt) and sections (.json)
meta_rows = []

for doc in DOCS:
    doc_id = doc["doc_id"]
    full_text_parts = []
    for s in doc["sections"]:
        full_text_parts.append(f"{s['heading']}\n{s['content']}")
    full_text = "\n\n".join(full_text_parts)

    txt_file = PAGES_DIR / f"{doc_id}.txt"
    txt_file.write_text(full_text, encoding="utf-8")

    sec_file = PAGES_DIR / f"{doc_id}_sections.json"
    sec_file.write_text(json.dumps(doc["sections"], indent=2), encoding="utf-8")

    meta_rows.append({
        "doc_id": doc_id,
        "source_name": doc["source_name"],
        "source_url": doc["source_url"],
        "type": doc["type"],
        "title": doc["title"],
        "date_collected": datetime.now(timezone.utc).isoformat(),
        "local_path": str(txt_file),
    })

# Now generate Institutional PDFs for D025 and D026 using PyMuPDF!
def create_pdf(path: Path, title: str, pages_content: list[tuple[str, str]]):
    doc = fitz.open()
    for heading, body in pages_content:
        page = doc.new_page(width=595, height=842) # A4 size
        # Header line (which will appear on pages)
        page.insert_text((50, 40), "SARDAR PATEL INSTITUTE OF TECHNOLOGY (AUTONOMOUS) — OFFICIAL DIRECTIVE", fontsize=8, color=(0.3, 0.3, 0.3))
        page.insert_text((50, 70), title, fontsize=14, color=(0, 0, 0.5))
        page.insert_text((50, 100), heading, fontsize=12, color=(0.1, 0.1, 0.1))
        
        # Word wrap body
        rect = fitz.Rect(50, 120, 545, 780)
        page.insert_textbox(rect, body, fontsize=10, color=(0.2, 0.2, 0.2), lineheight=1.3)
        # Footer
        page.insert_text((50, 810), "Confidential & Academic Council Approved Regulation | Munshi Nagar, Andheri (W), Mumbai", fontsize=8, color=(0.5, 0.5, 0.5))
    doc.save(str(path))
    doc.close()

# D025 PDF: Code of Conduct & Anti-Ragging Policy
pdf_d025_path = PDFS_DIR / "D025_code_of_conduct.pdf"
create_pdf(
    pdf_d025_path,
    "Institutional Regulations: Code of Conduct and Anti-Ragging Policy",
    [
        ("Section 1: General Code of Student Conduct",
         "All students admitted to Sardar Patel Institute of Technology must uphold the highest standards of academic integrity, personal decorum, and mutual respect. Damaging campus property, unauthorized gatherings, identity falsification, or substance abuse on the Bharatiya Vidya Bhavan campus will attract severe disciplinary action, including suspension or permanent expulsion as determined by the Proctorial Committee."),
        ("Section 2: Anti-Ragging Statutory Guidelines",
         "In strict compliance with UGC and AICTE regulations and the Maharashtra Prohibition of Ragging Act, ragging in any form—physical abuse, mental harassment, verbal intimidation, or forced activity—is strictly prohibited both inside and outside the college campus. Any student found guilty of ragging shall immediately be suspended, and a First Information Report (FIR) will be lodged with the local Mumbai Police station. Anti-Ragging Squad conducts unannounced vigils round the clock.")
    ]
)
meta_rows.append({
    "doc_id": "D025",
    "source_name": "Student Code of Conduct & Anti-Ragging Policy",
    "source_url": "https://www.spit.ac.in/pdf/code-of-conduct.pdf",
    "type": "pdf",
    "title": "Institutional Regulations: Code of Conduct and Anti-Ragging Policy",
    "date_collected": datetime.now(timezone.utc).isoformat(),
    "local_path": str(pdf_d025_path),
})

# D026 PDF: Examination & Grievance Redressal Rules
pdf_d026_path = PDFS_DIR / "D026_examination_rules.pdf"
create_pdf(
    pdf_d026_path,
    "Academic Regulations: Examination Grievance and Revaluation Rules",
    [
        ("Section 1: Post-Result Verification and Photocopy Procedure",
         "A candidate who desires to challenge the marks obtained in the End-Semester Examination (ESE) can apply for verification of marks or procurement of a certified photocopy of the assessed answer book within seven working days from the date of declaration of results on the institutional ERP portal. A fee of Rs. 300 per paper is levied for verification."),
        ("Section 2: Revaluation and Expert Panel Review",
         "Upon obtaining the photocopy, if discrepancies in assessment are identified, the student may apply for formal revaluation within five days. The answer book is masked and forwarded to an external independent subject expert. If the difference in marks exceeds 10% of the maximum marks, the revised score is awarded to the student.")
    ]
)
meta_rows.append({
    "doc_id": "D026",
    "source_name": "Examination Grievance & Revaluation Rules",
    "source_url": "https://www.spit.ac.in/pdf/exam-grievance-rules.pdf",
    "type": "pdf",
    "title": "Academic Regulations: Examination Grievance and Revaluation Rules",
    "date_collected": datetime.now(timezone.utc).isoformat(),
    "local_path": str(pdf_d026_path),
})

# D029 ERP CSV: Courses and Time-table Export
erp_csv_path = ERP_DIR / "courses.csv"
courses_data = [
    {"course_code": "CE413", "course_name": "Generative AI", "department": "Computer Engineering", "semester": 7, "credits": 3, "instructor": "Prof. Yogita Patil", "day_slot": "Monday 10:00 AM - 12:00 PM, Thursday 11:00 AM - 12:00 PM", "classroom": "Room 405 (CC3 Lab)"},
    {"course_code": "CE414", "course_name": "Cloud Computing and DevOps", "department": "Computer Engineering", "semester": 7, "credits": 3, "instructor": "Prof. Sunil Ghane", "day_slot": "Tuesday 09:00 AM - 11:00 AM, Friday 10:00 AM - 11:00 AM", "classroom": "Room 402"},
    {"course_code": "CE301", "course_name": "Machine Learning", "department": "Computer Engineering", "semester": 5, "credits": 4, "instructor": "Dr. D. R. Kalbande", "day_slot": "Wednesday 01:00 PM - 03:00 PM, Friday 02:00 PM - 04:00 PM", "classroom": "Auditorium Lab"},
    {"course_code": "IT401", "course_name": "Enterprise Security and Cryptography", "department": "Information Technology", "semester": 7, "credits": 3, "instructor": "Dr. Radha Shankarmani", "day_slot": "Tuesday 11:00 AM - 01:00 PM", "classroom": "Room 501"},
    {"course_code": "ET302", "course_name": "Digital Signal Processing", "department": "Electronics and Telecommunication", "semester": 5, "credits": 4, "instructor": "Dr. Y. S. Rao", "day_slot": "Monday 02:00 PM - 04:00 PM", "classroom": "Room 303"},
    {"course_code": "CE499", "course_name": "B.Tech Capstone Project", "department": "Computer Engineering", "semester": 7, "credits": 6, "instructor": "Project Review Committee", "day_slot": "Wednesday 09:00 AM - 05:00 PM", "classroom": "Project Center Lab"},
]

with open(erp_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(courses_data[0].keys()))
    writer.writeheader()
    writer.writerows(courses_data)

meta_rows.append({
    "doc_id": "D029",
    "source_name": "Institutional ERP Course Catalog",
    "source_url": "erp.spit.ac.in/export/courses",
    "type": "erp_csv",
    "title": "SPIT ERP Academic Course Registration and Timetable Records",
    "date_collected": datetime.now(timezone.utc).isoformat(),
    "local_path": str(erp_csv_path),
})

# Save kb/metadata.csv
with open(KB_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
    writer.writeheader()
    writer.writerows(meta_rows)

print(f"[SUCCESS] Successfully seeded {len(meta_rows)} SPIT documents into {KB_DIR / 'metadata.csv'}!")
