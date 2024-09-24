import heapq
# pip install geopy
from geopy.distance import geodesic


class Recipient:
    def __init__(self, id, blood_type, urgency, location):
        self.id = id
        self.blood_type = blood_type
        self.urgency = urgency
        self.location = location

    def __lt__(self, other):
        return self.urgency > other.urgency  # Higher urgency gets higher priority

class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None

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