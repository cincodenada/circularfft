enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

struct BracketedChunks<I> where I: Iterator
{
    state: State,
    min: I::Item,
    max: I::Item,
    size: I::Item,
    candidate: Option<I::Item>,
    source: I,
}
impl<I> BracketedChunks<I> where I: Iterator {
    fn shard(&self, val: I::Item) -> usize {
        (val/self.size).floor()
    }

    fn same_shard(&self, a: I::Item, b: I::Item) -> bool {
        self.shard(a) == self.shard(b)
    }
}

impl<I> Iterator for BracketedChunks<I>
where
    I: Iterator
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let candidate_shard = self.shard(self.candidate);
        let (cur, next, state) = match (self.state, self.candidate) {
            (State::Start, None) =>
                (Some(self.min), self.source.find(|Some(v)| v >= self.min), State::Passthrough),
            (State::Start, _) =>
                panic!("Invalid state, had candidate at start!"),
            (State::Passthrough, c) => match self.source.next() {
                Some(next) if self.shard(next) == candidate_shard =>
                    (self.candidate, next, State::Passthrough),
                Some(next) =>
                    (self.candidate, next, State::NewShard),
                // TODO: This assumes max is always the end of the last shard
                // which is probably true for my case but could be more general
                None => (self.candidate, self.max, State::Finish)
            },
            (State::NewShard, c) => 
                ((candidate_shard as Self::Item)*self.size, c, State::Passthrough),
            (Finish, _) => (self.candidate, None, State::Exhausted),
            (Exhausted, _) => None
        };
        self.next = next;
        self.state = state;
        cur
    }
}

trait Bracketed: Iterator {
    fn bracketed_chunks<T>(self, size: T, min: T, max: T) -> BracketedChunks<Self> where Self:Sized {
        BracketedChunks {
            min, max, size,
            state: State::Start,
            candidate: None,
            source: self
        }
    }
}

impl<I: Iterator> Bracketed for I {}
