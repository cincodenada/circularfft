struct BracketedChunks<I> where I: Iterator,
{
    min: I::Item,
    max: I::Item,
    size: I::Item,
    next: Some(I::Item),
    source: I,
}

impl<I> Iterator for BracketedChunks<I>
where
    I: Iterator
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match next {
            None => 
        }
        if(!next) {
            while let next = source.next() if next < min {}
            return min
        }
        if(next == max) {
            next = None
            return max
        }
        if(next) {
            let cur = next
            next = source.next();
            if(!next) { next = max }
            // TODO: Make check for > case = bad
            if((cur/size).floor() < (next/size).floor()) {
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
