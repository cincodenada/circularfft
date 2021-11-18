enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

struct BracketedChunks<I> where I: Iterator,
{
    state: State,
    min: I::Item,
    max: I::Item,
    size: I::Item,
    candidate: Some(I::Item),
    source: I,
}

impl<I> Iterator for BracketedChunks<I>
where
    I: Iterator
{
    type Item = I::Item;

    fn shard(&self, val: Self::Item) -> usize {
        (val/self.size).floor()
    }

    fn same_shard(&self, a: Self::Item, b: Self::Item) -> bool {
        self.shard(a) == self.shard(b)
    }

    fn next(&mut self) -> Option<Self::Item> {
        let candidate_shard = self.shard(self.candidate)
        let (cur, self.next, self.state) = match (self.state, self.candidate) {
            (State::Start, None) =>
                (Some(min), source.find(|Some(v)| v >= min), State::Passthrough),
            (State::Start, _) =>
                panic!("Invalid state, had candidate at start!"),
            (State::Passthrough, c) => match source.next() {
                Some(next) if self.shard(next) == candidate_shard =>
                    (candidate, next, State::Passthrough),
                Some(next) =>
                    (candidate, next, State::NewShard)
                // TODO: This assumes max is always the end of the last shard
                // which is probably true for my case but could be more general
                None => (candidate, max, State::Finish)
            },
            (State::NewShard, c) => 
                ((candidate_shard as Self::Item)*size, c, State::Passthrough),
            (Finish, _) => (candidate, None, State::Exhausted),
            (Exhausted, _) => None
        };

        cur
    }
}

trait Bracketed: Iterator {
    fn bracketed_chunks<T>(self, size: T, min: T, max: T) -> BracketedChunks<Self>
    where Self::Item: T
    {
        BracketedChunks {
            min, max, size,
            state: State::Start,
            candidate: None,
            source: self
        }
    }
}

impl<I: Iterator> Bracketed for I {}
