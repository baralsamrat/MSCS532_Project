


```diff
- Project Phase 1 : September 08, 2024
- Project Phase 2 : September 23, 2024
+ Project Phase 3 : Current
```
> Samrat Baral
> Algorithm and Data Structures

# Organ Matching and Donation Networks


## Project Phase 1 Deliverable 1: Data Structure Design and Implementation
## Application Context
Organ donation is a crucial area in healthcare, potentially saving countless lives. The challenge is to efficiently match donors with recipients based on various criteria, including blood type, HLA matching, geographic proximity, and urgency. An effective algorithm can leverage real-time data for optimal matching, ensuring timely transplants as the demand for organs increases.

## Design
The architecture of the donation matching system needs to prioritize fast lookups and manage urgency and optimal matches based on location. 

### Donor and Recipient Requirements
- **Authentication**: Users must verify their identities using medical records, adhering to HIPAA regulations and relevant laws to ensure secure access to health information.

### Potential Input Requirements
1. Blood Type
2. Human Leukocyte Antigen (HLA)
3. Geographic Proximity
4. Urgency
5. HCT levels (optional)

### Potential Data Structures
1. **Priority Queue (Heap)**
   - **Insertion/Deletion**: O(log n)
   - Consider using a Fibonacci heap for frequent insertions with an amortized time of O(1).

2. **Hash Table/Maps**
   - Maps blood types to donor/recipient lists.
   - **Time Complexity**: O(1)

3. **Binary Tree (Balanced Tree or Red-Black Tree)**
   - For storing user information and urgency levels.
   - **Time Complexity**: O(log n)

4. **Graphs**
   - Use Dijkstra’s algorithm for finding optimal paths based on geographic data.
   - **Time Complexity**: O(log n) + O(V + E log V)

## Key Ideas on Data Structures

### Priority Queue
```python
class Recipient:
    def __init__(self, id, blood_type, urgency):
        self.id = id
        self.blood_type = blood_type
        self.urgency = urgency

    def __lt__(self, other):
        return self.urgency > other.urgency  # Higher urgency gets higher priority

class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None
```

### Hash Table
```python
class Donor:
    def __init__(self, id, blood_type, hla_type):
        self.id = id
        self.blood_type = blood_type
        self.hla_type = hla_type

class DonorHashMap:
    def __init__(self):
        self.hash_table = {blood_type: [] for blood_type in ["A", "B", "AB", "O"]}

    def add_donor(self, donor):
        self.hash_table[donor.blood_type].append(donor)

    def get_donors_by_blood_type(self, blood_type):
        return self.hash_table.get(blood_type, [])
```

### Binary Tree (AVL Tree)
```python
class AVLNode:
    def __init__(self, recipient):
        self.recipient = recipient
        self.height = 1
        self.left = None
        self.right = None

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, recipient):
        # Insertion logic goes here, including balancing the tree
        pass
```

## Architecture: Estimation and Potential Requirements
1. **Donor and Recipient Entry**: Incorporate privacy measures and smart contracts.
2. **Matching Criteria**: Implement real-time matching, considering unmatched donors.
3. **Network Security**: Leverage blockchain technology for secure data handling.
4. **Kidney Paired Exchange (KPE)**: Utilize KPE algorithms to optimize organ exchanges.
5. **Compliance**: Adhere to ethical and regulatory standards.
6. **Machine Learning**: Implement predictive models for optimal donor selection.

## Python Library Implementation
- `heapq`: For priority queues.
- `sortedcontainers`: For sorted data structures.
- `bintree`: For balanced trees.
- `NetworkX`: For graph-related operations.
- `SQLAlchemy`: For database interactions.
- `scikit-learn`: For machine learning implementations.
- `NumPy` and `Pandas`: For data manipulation.
- `schedule`: For scheduling periodic tasks.

## Potential Challenges and Limitations
- **Scalability**: Addressing hash table collisions and ensuring efficient lookups.
- **Geographic Limitations**: Potentially utilizing AI models for routing.
- **Real-Time Updates**: Managing updates without excessive computational costs.
- **Healthcare Compliance**: Navigating evolving regulations.
- **Data Sources**: Automating data updates from various APIs while ensuring data integrity.

## Python API Use Case
```python
import requests
import json

api_endpoint = "https://api.example.com/donors"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

def update_data():
    response = requests.get(api_endpoint, headers=headers)
    if response.status_code == 200:
        data = response.json()
        save_to_database(data)
    else:
        print(f"Failed to fetch data: {response.status_code}")

def save_to_database(data):
    # Implement database saving logic here
    pass

if __name__ == "__main__":
    update_data()
    # Schedule the update function to run periodically
```



### Demonstration of Key Operations
The provided test script demonstrates how to add users, establish product relationships, and generate recommendations. Running this script will output the top product recommendation based on the scores assigned.

```python

import heapq
from collections import defaultdict

# Recipient class representing the recipient details
class Recipient:
    def __init__(self, id, blood_type, urgency):
        self.id = id
        self.blood_type = blood_type
        self.urgency = urgency

    def __lt__(self, other):
        return self.urgency > other.urgency  # Higher urgency gets higher priority

# Priority queue for managing recipient urgency
class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None

# Donor class representing the donor details
class Donor:
    def __init__(self, id, blood_type, hla_type):
        self.id = id
        self.blood_type = blood_type
        self.hla_type = hla_type

# Hash table to manage donors by blood type
class DonorHashMap:
    def __init__(self):
        self.hash_table = defaultdict(list)

    def add_donor(self, donor):
        self.hash_table[donor.blood_type].append(donor)

    def get_donors_by_blood_type(self, blood_type):
        return self.hash_table.get(blood_type, [])

# Simple matching algorithm based on blood type and urgency
def match_donors_to_recipients(donor_map, urgency_queue):
    matches = []
    while urgency_queue.heap:
        recipient = urgency_queue.get_highest_priority()
        donors = donor_map.get_donors_by_blood_type(recipient.blood_type)
        
        if donors:
            matched_donor = donors.pop(0)  # Get the first available donor
            matches.append((recipient.id, matched_donor.id))
        else:
            print(f"No suitable donor found for recipient {recipient.id} with blood type {recipient.blood_type}.")
    
    return matches

# Test script demonstrating functionality
if __name__ == "__main__":
    # Initialize data structures
    donor_map = DonorHashMap()
    urgency_queue = UrgencyQueue()

    # Adding donors
    donor1 = Donor(id="donor1", blood_type="A", hla_type="HLA-A1")
    donor2 = Donor(id="donor2", blood_type="B", hla_type="HLA-B1")
    donor3 = Donor(id="donor3", blood_type="A", hla_type="HLA-A2")
    donor_map.add_donor(donor1)
    donor_map.add_donor(donor2)
    donor_map.add_donor(donor3)

    # Adding recipients with varying urgency
    recipient1 = Recipient(id="recipient1", blood_type="A", urgency=5)
    recipient2 = Recipient(id="recipient2", blood_type="B", urgency=10)
    recipient3 = Recipient(id="recipient3", blood_type="A", urgency=8)
    urgency_queue.add_recipient(recipient1)
    urgency_queue.add_recipient(recipient2)
    urgency_queue.add_recipient(recipient3)

    # Matching donors to recipients
    matches = match_donors_to_recipients(donor_map, urgency_queue)
    for recipient_id, donor_id in matches:
        print(f"Matched recipient {recipient_id} with donor {donor_id}.")

```

Output
``` bash
    python3 main.py 
    Matched recipient recipient2 with donor donor2.
    Matched recipient recipient3 with donor donor1.
    Matched recipient recipient1 with donor donor3.
```

### Challenges Encountered
- **Integration of Data Structures**: Ensuring the user profiles, product relationships, and recommendations work cohesively posed initial challenges. I tackled this by defining clear interfaces for each data structure, allowing easy data flow.
  
- **Scoring System**: Developing a straightforward and effective scoring mechanism for recommendations required careful thought. I implemented a basic scoring method, which can be refined later.

### Solutions Implemented
- **Modular Design**: Each data structure was designed with clear responsibilities, making future expansions easier.
- **Error Handling**: Basic error handling was included to manage cases such as duplicate users or missing product relationships.

### Next Steps
1. **Enhance User Interaction Tracking**: Implement a more sophisticated method for tracking user interactions and preferences to improve recommendation accuracy.
2. **Advanced Recommendation Algorithms**: Explore collaborative filtering and content-based filtering techniques for generating more personalized recommendations.
3. **User Interface Development**: Develop a simple UI to facilitate user interaction with the recommendation system.
4. **Testing and Validation**: Conduct extensive testing to validate functionality, performance, and scalability as the application grows.

## GitHub Repository
The full codebase, including the implementation of data structures and the test script, is available in the following GitHub repository: [MSCS532_Project](https://github.com/baralsamrat/MSCS532_Project)


This example includes a priority queue for recipients, a hash table for donors, and a basic structure for geographical matching using a simplified approach.

### Project Phase 2 Deliverable 2: Proof of Concept Implementation

#### 1. Recipient Class and Priority Queue

```python
import heapq

class Recipient:
    def __init__(self, id, blood_type, urgency, location):
        self.id = id
        self.blood_type = blood_type
        self.urgency = urgency
        self.location = location #new

    def __lt__(self, other):
        return self.urgency > other.urgency  # Higher urgency gets higher priority

class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None
```

#### 2. Donor Class and Hash Table

```python
class Donor:
    def __init__(self, id, blood_type, hla_type, location):
        self.id = id
        self.blood_type = blood_type
        self.hla_type = hla_type
        self.location = location

class DonorHashMap:
    def __init__(self):
        self.hash_table = {blood_type: [] for blood_type in ["A", "B", "AB", "O"]}

    def add_donor(self, donor):
        self.hash_table[donor.blood_type].append(donor)

    def get_donors_by_blood_type(self, blood_type):
        return self.hash_table.get(blood_type, [])
```

#### 3. Simple Geographical Matching Function

#### Installation: [GeoPy](https://geopy.readthedocs.io/en/stable/#installation)
```bash
pip install geopy
```

```python
from geopy.distance import geodesic

def find_best_match(recipient, donors):
    best_match = None
    best_distance = float('inf')
    
    for donor in donors:
        if donor.blood_type == recipient.blood_type:  # Check blood type compatibility
            distance = geodesic(recipient.location, donor.location).miles
            if distance < best_distance:
                best_distance = distance
                best_match = donor
    
    return best_match
```

#### 4. Proof of Concept Usage

```python
if __name__ == "__main__":
    # Create a queue for recipients
    recipient_queue = UrgencyQueue()

    # Add recipients
    recipient1 = Recipient("rec1", "A", 5, (40.7128, -74.0060))  # New York
    recipient2 = Recipient("rec2", "B", 10, (34.0522, -118.2437))  # Los Angeles
    recipient_queue.add_recipient(recipient1)
    recipient_queue.add_recipient(recipient2)

    # Create a hash map for donors
    donor_map = DonorHashMap()

    # Add donors
    donor1 = Donor("don1", "A", "HLA1", (41.8781, -87.6298))  # Chicago
    donor2 = Donor("don2", "B", "HLA2", (34.0522, -118.2437))  # Los Angeles
    donor_map.add_donor(donor1)
    donor_map.add_donor(donor2)

    # Process matching for the highest priority recipient
    highest_priority_recipient = recipient_queue.get_highest_priority()
    donors_of_same_blood_type = donor_map.get_donors_by_blood_type(highest_priority_recipient.blood_type)
    best_match = find_best_match(highest_priority_recipient, donors_of_same_blood_type)

    if best_match:
        print(f"Best match for {highest_priority_recipient.id}: Donor ID {best_match.id} at location {best_match.location}")
    else:
        print(f"No suitable donor found for {highest_priority_recipient.id}.")
```

### Explanation
- **Recipient Class**: Holds information about each recipient, including urgency and location.
- **UrgencyQueue Class**: Implements a priority queue to manage recipients based on urgency.
- **Donor Class**: Stores donor information, including blood type and location.
- **DonorHashMap Class**: Uses a hash table to organize donors by blood type.
- **find_best_match Function**: Compares recipients and donors based on blood type and geographic distance, returning the closest suitable donor.

### Future Enhancements
- **Integration with Databases**: Store recipient and donor data in a database for persistence.
- **Machine Learning**: Implement predictive matching algorithms based on historical data.
- **Real-time Updates**: Use webhooks or similar methods to update donor/recipient information in real time.
- **Advanced Geolocation**: Enhance the geographical matching function with more sophisticated routing algorithms.

Feel free to adapt and expand this code base according to your project needs!
### Phase 3: Optimization, Scaling, and Final Evaluation 

1. Optimization of Data Structures
Analyze performance and identify inefficiencies.
Implement optimizations like caching frequently accessed user profiles.

2. Scaling for Large Datasets
Modify implementations to manage larger datasets effectively, ensuring acceptable performance levels.

3. Advanced Testing and Validation
Develop comprehensive test cases and perform stress testing to evaluate robustness.

4. Final Evaluation and Performance Analysis
Compare the final implementation with the initial proof of concept, discussing strengths, limitations, and areas for improvement.


Here are more detailed future ideas on how to implement the future enhancements for your **Organ Matching and Donation Network**:

### 3.1 Integration with Databases

**Objective:** Store recipient and donor data for persistence, enabling easy retrieval and management.

**Implementation Steps:**
- **Choose a Database:** Select a relational database (e.g., PostgreSQL, MySQL) or a NoSQL database (e.g., MongoDB) based on your data structure and querying needs.
  
- **Database Models:**
  Define models for Donor and Recipient, using an ORM like SQLAlchemy for relational databases or a library like PyMongo for NoSQL.

  ```python
  from sqlalchemy import create_engine, Column, String, Integer, Float
  from sqlalchemy.ext.declarative import declarative_base
  from sqlalchemy.orm import sessionmaker

  Base = declarative_base()

  class Donor(Base):
      __tablename__ = 'donors'
      id = Column(String, primary_key=True)
      blood_type = Column(String)
      hla_type = Column(String)
      location_lat = Column(Float)
      location_lon = Column(Float)

  class Recipient(Base):
      __tablename__ = 'recipients'
      id = Column(String, primary_key=True)
      blood_type = Column(String)
      urgency = Column(Integer)
      location_lat = Column(Float)
      location_lon = Column(Float)

  # Set up the database
  engine = create_engine('sqlite:///organ_donation.db')
  Base.metadata.create_all(engine)
  Session = sessionmaker(bind=engine)
  ```

- **Data Access Methods:** Implement functions to add, retrieve, and update donor/recipient records.

### 3.2 Machine Learning

**Objective:** Use historical data to improve matching accuracy through predictive algorithms.

**Implementation Steps:**
- **Data Collection:** Gather historical donor and recipient data, including successful matches and outcomes.
  
- **Feature Engineering:** Identify relevant features (e.g., blood type, urgency, location distance) that influence successful matches.

- **Model Training:**
  - Use libraries like scikit-learn or TensorFlow to create and train models (e.g., logistic regression, decision trees) on your dataset.
  
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier

  # Sample feature and target arrays
  X = ...  # Features: blood type, distance, urgency
  y = ...  # Target: match success

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  ```

- **Predictive Matching:** Integrate the model into your matching logic to prioritize matches based on predicted success rates.

### 3.3 Real-time Updates

**Objective:** Ensure that donor and recipient information is up-to-date without manual intervention.

**Implementation Steps:**
- **Webhooks:** Set up webhooks to listen for changes in external data sources (e.g., hospitals, health organizations).

- **API Integration:**
  - Use libraries like `requests` to fetch real-time updates from APIs.
  
  ```python
  import requests

  def fetch_donor_data():
      response = requests.get('https://api.example.com/donors')
      if response.status_code == 200:
          data = response.json()
          # Update the database with new data
      else:
          print(f"Error fetching data: {response.status_code}")
  ```

- **Scheduled Tasks:** Use a task scheduler (e.g., `schedule` library, Celery) to periodically check for updates.

### 3.4 Advanced Geolocation

**Objective:** Enhance geographic matching with sophisticated algorithms.

**Implementation Steps:**
- **Routing Algorithms:** Implement algorithms like Dijkstra’s or A* for more accurate pathfinding based on real traffic data.

- **Geospatial Libraries:** Use libraries like `geopy` or `Shapely` to perform complex geospatial operations.

- **Distance Calculation:**
  - Improve distance calculations by factoring in real-world conditions like traffic patterns.

  ```python
  from geopy.distance import geodesic

  def calculate_distance(location1, location2):
      return geodesic(location1, location2).miles
  ```

- **Integration with Mapping APIs:** Use services like Google Maps API to retrieve real-time data about distances and estimated travel times.

## References
1. Igboanusi, I. S., Nnadiekwe, C. A., Ogbede, J. U., Kim, D.-S., & Lensky, A. (2024). BOMS:blockchain-enabled organ matching system. Scientific Reports, 14(1), 1–13. https://doi.org/10.1038/s41598-024-66375-5
2. Al-Thnaibat, M. H., Balaw, M. K., Al-Aquily, M. K., Ghannam, R. A., Mohd, O. B., Alabidi, F., Alabidi, S., Hussein, F., & Rawashdeh, B. (2024). Addressing Kidney Transplant Shortage: The Potential of Kidney Paired Exchanges in Jordan. Journal of Transplantation, 2024, 1–8.https://doi.org/10.1155/2024/4538034
3. Cloutier, M., Grégoire, Y., Choucha, K., Amja, A.-M., & Lewin, A. (2021). Prediction of donation return rate in young donors using machine-learning models. ISBT Science Series, 16(1), 119–126. https://doi.org/10.1111/voxs.12618
4. Connor, J. P., Raife, T., & Medow, J. E. (2018). Outcomes of red blood cell transfusions prescribed in organ donors by the Digital Intern, an electronic decision support algorithm. Transfusion, 58(2), 366–371. https://doi.org/10.1111/trf.14424
Other References 
5. ONC’s Cures Act Final Rule | HealthIT.gov. (2022, August 31). Www.healthit.gov. 
https://www.healthit.gov/topic/oncs-cures-act-final-rule
6. PacktPublishing. “Packtpublishing/Mastering-Geospatial-Analysis-with-Python: Mastering Geospatial Analysis with Python, Published by Packt.” GitHub, https://github.com/PacktPublishing/Mastering-Geospatial-Analysis-with-Python?tab=readme-ov-file. Accessed 23 Sept. 2024. 
7. “Pattern Recognition and Machine Learning - Free Computer, Programming, Mathematics, Technical Books, Lecture Notes and Tutorials.” FreeComputerBooks, https://freecomputerbooks.com/Pattern-Recognition-and-Machine-Learning.html. Accessed 23 Sept. 2024. 
8. Pattern Recognition and Machine Learning, https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf. Accessed 24 Sept. 2024. 
9. “Sqlalchemy.” PyPI, https://pypi.org/project/SQLAlchemy/. Accessed 23 Sept. 2024.
10. Sundriyal, Vaibhav, and Masha Sosonkina. “Runtime Energy Savings Based on Machine Learning Models for Multicore Applications.” SCIRP, Scientific Research Publishing, 9 June 2022, https://www.scirp.org/journal/paperinformation?paperid=118212. 
11. “Welcome to GeoPy’s Documentation!.” Welcome to GeoPy’s Documentation! - GeoPy 2.4.1 Documentation, https://geopy.readthedocs.io/en/stable/#installation. Accessed 23 Sept. 2024. 

----