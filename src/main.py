from dataclasses import dataclass
import random
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@dataclass
class State:
    ball_location: int  # 0 = basket, 1 = box
    sally_in_room: int  # 0 = absent, 1 = present
    sally_belief: int   # 0 = believes basket, 1 = believes box

class SallyAnneEnvironment:

    ACTION_MOVE_TO_BASKET = 0
    ACTION_MOVE_TO_BOX    = 1
    ACTION_TOGGLE_SALLY   = 2

    def __init__(self):
        self.state = None
        self.reset()
    
    def reset(self):
        ball = random.randint(0, 1)
        sally = random.randint(0, 1)
        self.state = State(
            ball_location=ball,
            sally_in_room=sally,
            sally_belief=ball if sally else random.randint(0, 1)
        )
        return self.state
    
    def step(self, action: int) -> State:
        if action == self.ACTION_MOVE_TO_BASKET:
            self.state.ball_location = 0
        elif action == self.ACTION_MOVE_TO_BOX:
            self.state.ball_location = 1
        elif action == self.ACTION_TOGGLE_SALLY:
            self.state.sally_in_room = 1 - self.state.sally_in_room
        
        if self.state.sally_in_room:
            self.state.sally_belief = self.state.ball_location
        
        return self.state
    

BELIEF_TEMPLATES = {
    0: [
        "Sally believes the ball is in the basket",
        "Sally thinks the ball is in the basket",
        "Sally expects to find the ball in the basket",
    ],
    1: [
        "Sally believes the ball is in the box",
        "Sally thinks the ball is in the box",
        "Sally expects to find the ball in the box",
    ],
}

NOVEL_BELIEF_TEMPLATES = {
    0: [
        "As far as Sally knows, the basket holds the ball",
        "Sally would look in the basket first",
        "Sally is convinced the ball never left the basket",
    ],
    1: [
        "As far as Sally knows, the box holds the ball",
        "Sally would look in the box first",
        "Sally is convinced the ball never left the box",
    ],
}

def belief_to_language(sally_belief: int) -> str:
    return random.choice(BELIEF_TEMPLATES[sally_belief])

def generate_episode(environment, num_steps=5, action_weights=None):
    episode = []
    state   = environment.reset()

    for _ in range(num_steps):
        if action_weights is None:
            action = random.randint(0, 2)
        else:
            action = random.choices([0, 1, 2], weights=action_weights, k=1)[0]
        current_state = State(
            ball_location=state.ball_location,
            sally_in_room=state.sally_in_room,
            sally_belief=state.sally_belief
        )
        description = belief_to_language(current_state.sally_belief)
        next_state = environment.step(action)
        episode.append((current_state, description, action, State(
            ball_location=next_state.ball_location,
            sally_in_room=next_state.sally_in_room,
            sally_belief=next_state.sally_belief,
        )))
        state = next_state
    
    return episode

def generate_dataset(num_episodes=1000, num_steps=5, action_weights=None):
    environment = SallyAnneEnvironment()
    dataset = []
    for _ in range(num_episodes):
        episode = generate_episode(environment, num_steps, action_weights)
        dataset.append(episode)
    return dataset

def restyle_descriptions(dataset, templates):
    restyled = []
    for episode in dataset:
        new_episode = [
            (current_state, random.choice(templates[current_state.sally_belief]), action, next_state)
            for current_state, _, action, next_state in episode
        ]
        restyled.append(new_episode)
    return restyled

def flatten(dataset):
    return [sample for episode in dataset for sample in episode]

def combo_key(current_state, action):
    return (current_state.ball_location, current_state.sally_in_room, current_state.sally_belief, action)

class BeliefTransitionModel(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=32, num_actions=3):
        super().__init__()
        input_dim = 3 + embedding_dim + num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, state: torch.Tensor, belief_emb: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, belief_emb, action_vec], dim=-1)
        return self.net(x)


def prepare_tensors(dataset, encoder):
    states, embeddings, actions, targets = [], [], [], []

    for episode in dataset:
        for current_state, description, action, next_state in episode:
            state_vec = torch.tensor([
                current_state.ball_location,
                current_state.sally_in_room,
                current_state.sally_belief
            ], dtype=torch.float32)

            embedding  = encoder.encode(description, convert_to_tensor=True)
            action_vec = nn.functional.one_hot(torch.tensor(action), num_classes=3).float()
            target     = torch.tensor(next_state.sally_belief, dtype=torch.long)

            states.append(state_vec)
            embeddings.append(embedding)
            actions.append(action_vec)
            targets.append(target)

    return (
        torch.stack(states),
        torch.stack(embeddings),
        torch.stack(actions),
        torch.stack(targets)
    )

def train(model, dataset, encoder, epochs=15, batch_size=64, lr=1e-3):
    model.to(device)
    states, embeddings, actions, targets = prepare_tensors(dataset, encoder)
    loader = DataLoader(
        TensorDataset(states, embeddings, actions, targets),
        batch_size=batch_size,
        shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for state_batch, emb_batch, action_batch, target_batch in loader:
            state_batch  = state_batch.to(device)
            emb_batch    = emb_batch.to(device)
            action_batch = action_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            logits = model(state_batch, emb_batch, action_batch)
            loss   = loss_fn(logits, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}  loss: {total_loss/len(loader)}")

def evaluate_and_report(model, dataset, encoder, label):
    model.eval()
    states, embeddings, actions, targets = prepare_tensors(dataset, encoder)
    states     = states.to(device)
    embeddings = embeddings.to(device)
    actions    = actions.to(device)
    targets    = targets.to(device)

    with torch.no_grad():
        logits = model(states, embeddings, actions)
        preds  = logits.argmax(dim=-1)

        sally_in  = states[:, 1].bool()
        sally_out = ~sally_in

        acc_in    = (preds[sally_in]  == targets[sally_in]).float().mean().item()  if sally_in.any()  else float("nan")
        acc_out   = (preds[sally_out] == targets[sally_out]).float().mean().item() if sally_out.any() else float("nan")
        acc_total = (preds == targets).float().mean().item()

    print(f"\n[{label}]")
    print(f"Accuracy when Sally present: {acc_in    * 100:.1f}%")
    print(f"Accuracy when Sally absent:  {acc_out   * 100:.1f}%")
    print(f"Overall accuracy:            {acc_total * 100:.1f}%")


if __name__ == "__main__":
    dataset = generate_dataset(num_episodes=1000, num_steps=5)

    basket_count = 0
    box_count = 0
    for episode in dataset:
        for current_state, description, action, next_state in episode:
            if next_state.sally_belief == 0:
                basket_count += 1
            else:
                box_count += 1
    for episode in dataset[:3]:
        for current_state, description, action, next_state in episode:
            print(f"ball={current_state.ball_location} sally_in={current_state.sally_in_room} belief={current_state.sally_belief} action={action} -> next_belief={next_state.sally_belief}")
        print("***")
    total = basket_count + box_count
    print(f"Basket: {basket_count/total*100:.1f}%")
    print(f"Box:    {box_count/total*100:.1f}%")

    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data  = dataset[split:]

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    model = BeliefTransitionModel()
    train(model, train_data, encoder, epochs=15, batch_size=64, lr=1e-3)

    evaluate_and_report(model, test_data, encoder, "In-distribution test set")

    print("\n*** Generalization Tests ***")

    combo_pool   = flatten(generate_dataset(num_episodes=1000, num_steps=5))
    unique_combos = sorted(set(combo_key(s[0], s[2]) for s in combo_pool))
    held_out_combos = set(unique_combos[::4])

    combo_train_samples = [s for s in combo_pool if combo_key(s[0], s[2]) not in held_out_combos]
    combo_test_samples  = [s for s in combo_pool if combo_key(s[0], s[2]) in held_out_combos]

    combo_model = BeliefTransitionModel()
    train(combo_model, [combo_train_samples], encoder, epochs=15, batch_size=64, lr=1e-3)
    evaluate_and_report(
        combo_model, [combo_test_samples], encoder,
        f"Unseen (state, action) combos  [{len(held_out_combos)} combos held out, {len(combo_test_samples)} samples]"
    )

    novel_test_data = restyle_descriptions(test_data, NOVEL_BELIEF_TEMPLATES)
    evaluate_and_report(model, novel_test_data, encoder, "Novel belief phrasing (unseen sentence templates)")

    long_dataset = generate_dataset(num_episodes=200, num_steps=20)
    evaluate_and_report(model, long_dataset, encoder, "Longer episodes (20 steps vs. 5 trained)")

    skewed_dataset = generate_dataset(num_episodes=200, num_steps=5, action_weights=[0.05, 0.05, 0.9])
    evaluate_and_report(model, skewed_dataset, encoder, "Skewed action distribution (90% toggle)")