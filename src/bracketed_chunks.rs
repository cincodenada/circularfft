struct BracketedChunks<I> where I: Iterator,
{
    min: I::Item,
    max: I::Item,
    size: I::Item,
    prev: I::Item,
    source: I,
}

impl<I> Iterator for BracketedChunks<I>
where
    I: Iterator
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if(!prev) {
            while let prev = source.next() if prev < min {}
            return min
        }
        if(prev == max) {
            prev = None
            return max
        }
        if(prev) {
            let cur = prev
            prev = source.next();
            if(!prev) { prev = max }
            // TODO: Make check for > case = bad
            if((cur/size).floor() < (prev/size).floor()) {
                return (cur/size).floor()
            } else {
                return cur
            }
        } else {
            return None
        }
    }
}

trait Bracketed: Iterator {
    fn bracketed_chunks<T>(self, min: T, max: T) -> BracketedChunks<Self>
    where Self::Item: T
    {
        BracketedChunks {
            seen: HashSet::new(),
            underlying: self,
        }
    }
}

impl<I: Iterator> Bracketed for I {}
