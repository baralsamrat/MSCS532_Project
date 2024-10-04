import heapq
from collections import defaultdict
from geopy.distance import geodesic
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, Column, String, Integer, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Set up the database
Base = declarative_base()
engine = create_engine('sqlite:///organ_donation.db')
Session = sessionmaker(bind=engine)
session = Session()

# Donor and Recipient Classes
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

Base.metadata.create_all(engine)

# Priority Queue to manage recipient urgency
class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None

# Function to calculate the best donor match based on blood type and location
def find_best_match(recipient, donors):
    best_match = None
    best_distance = float('inf')

    for donor in donors:
        distance = geodesic((recipient.location_lat, recipient.location_lon), (donor.location_lat, donor.location_lon)).miles
        if distance < best_distance:
            best_distance = distance
            best_match = donor

    return best_match

# Machine Learning Model for predictive matching
data = pd.read_csv('organ_donation_data.csv')
features = data[['blood_type_donor', 'blood_type_recipient', 'urgency']]
labels = data['match_success']

# Train the predictive model
model = RandomForestClassifier()
model.fit(features, labels)

# Function to match recipients with donors and predict match success
def match_recipients_to_donors(recipient_queue):
    while recipient_queue.heap:
        recipient = recipient_queue.get_highest_priority()
        donors = session.query(Donor).filter(Donor.blood_type == recipient.blood_type).all()

        if donors:
            best_match = find_best_match(recipient, donors)
            if best_match:
                prediction = model.predict([[best_match.blood_type, recipient.blood_type, recipient.urgency]])
                if prediction[0] == 1:
                    print(f"Matched recipient {recipient.id} with donor {best_match.id}.")
                else:
                    print(f"No suitable donor found for recipient {recipient.id}.")
        else:
            print(f"No donors available for recipient {recipient.id}.")

# Add sample recipients and donors
def add_sample_data():
    # Add sample donors to the database
    donor1 = Donor(id="don1", blood_type="A", hla_type="HLA1", location_lat=41.8781, location_lon=-87.6298)  # Chicago
    donor2 = Donor(id="don2", blood_type="B", hla_type="HLA2", location_lat=34.0522, location_lon=-118.2437)  # Los Angeles
    donor3 = Donor(id="don3", blood_type="A", hla_type="HLA3", location_lat=40.7128, location_lon=-74.0060)  # New York
    
    session.add(donor1)
    session.add(donor2)
    session.add(donor3)
    session.commit()

    # Add sample recipients to the queue
    recipient1 = Recipient(id="rec1", blood_type="A", urgency=5, location_lat=40.7128, location_lon=-74.0060)  # New York
    recipient2 = Recipient(id="rec2", blood_type="B", urgency=10, location_lat=34.0522, location_lon=-118.2437)  # Los Angeles
    recipient3 = Recipient(id="rec3", blood_type="A", urgency=8, location_lat=41.8781, location_lon=-87.6298)  # Chicago

    recipient_queue = UrgencyQueue()
    recipient_queue.add_recipient(recipient1)
    recipient_queue.add_recipient(recipient2)
    recipient_queue.add_recipient(recipient3)

    # Match recipients to donors
    match_recipients_to_donors(recipient_queue)

# Main Execution
if __name__ == "__main__":
    add_sample_data()
