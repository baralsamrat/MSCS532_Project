import heapq
from geopy.distance import geodesic
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///organ_donation.db')
Session = sessionmaker(bind=engine)
session = Session()

class Donor(Base):
    __tablename__ = 'donors'
    id = Column(String, primary_key=True)
    blood_type = Column(String)
    hla_type = Column(String)
    location = Column(String)

class Recipient(Base):
    __tablename__ = 'recipients'
    id = Column(String, primary_key=True)
    blood_type = Column(String)
    urgency = Column(Integer)
    location = Column(String)

Base.metadata.create_all(engine)

# Smart Contract Simulation
class OrganDonation:
    def __init__(self):
        self.donors = {}
        self.recipients = {}

    def add_donor(self, donor_id, blood_type, hla_type, location):
        self.donors[donor_id] = Donor(id=donor_id, blood_type=blood_type, hla_type=hla_type, location=location)

    def add_recipient(self, recipient_id, blood_type, urgency, location):
        self.recipients[recipient_id] = Recipient(id=recipient_id, blood_type=blood_type, urgency=urgency, location=location)

    def match_donor(self, recipient_id):
        recipient = self.recipients[recipient_id]
        best_match = None
        best_distance = float('inf')
        
        for donor_id, donor in self.donors.items():
            if donor.blood_type == recipient.blood_type:  # Check blood type compatibility
                distance = geodesic(recipient.location, donor.location).miles
                if distance < best_distance:
                    best_distance = distance
                    best_match = donor

        return best_match

# Priority Queue for managing recipient urgency
class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None

# Machine Learning Model Preparation
def prepare_ml_model(data_file):
    data = pd.read_csv(data_file)
    features = data[['blood_type_donor', 'blood_type_recipient', 'urgency']]
    labels = data['match_success']
    model = RandomForestClassifier()
    model.fit(features, labels)
    return model

def predict_match(model, donor_info):
    return model.predict(pd.DataFrame([donor_info]))

# Main Execution Flow
if __name__ == "__main__":
    # Create tables
    Base.metadata.create_all(engine)

    # Initialize the organ donation system
    donation_system = OrganDonation()
    recipient_queue = UrgencyQueue()

    # Add recipients
    recipient1 = Recipient(id="rec1", blood_type="A", urgency=5, location=(40.7128, -74.0060))  # New York
    recipient2 = Recipient(id="rec2", blood_type="B", urgency=10, location=(34.0522, -118.2437))  # Los Angeles
    recipient_queue.add_recipient(recipient1)
    recipient_queue.add_recipient(recipient2)

    # Add donors
    donor1 = Donor(id="don1", blood_type="A", hla_type="HLA1", location=(41.8781, -87.6298))  # Chicago
    donor2 = Donor(id="don2", blood_type="B", hla_type="HLA2", location=(34.0522, -118.2437))  # Los Angeles
    donation_system.add_donor(donor1.id, donor1.blood_type, donor1.hla_type, donor1.location)
    donation_system.add_donor(donor2.id, donor2.blood_type, donor2.hla_type, donor2.location)

    # Prepare Machine Learning Model
    model = prepare_ml_model('organ_donation_data.csv')  # Ensure this CSV is present

    # Matching process
    while recipient_queue.heap:
        highest_priority_recipient = recipient_queue.get_highest_priority()
        best_match = donation_system.match_donor(highest_priority_recipient.id)

        if best_match:
            donor_info = {
                'blood_type_donor': best_match.blood_type,
                'blood_type_recipient': highest_priority_recipient.blood_type,
                'urgency': highest_priority_recipient.urgency
            }
            match_prediction = predict_match(model, donor_info)
            if match_prediction[0] == 1:  # If the prediction is successful
                print(f"Matched recipient {highest_priority_recipient.id} with donor {best_match.id}.")
            else:
                print(f"No suitable donor found for recipient {highest_priority_recipient.id}.")
        else:
            print(f"No donors available for recipient {highest_priority_recipient.id}.")
