from rerankers.models.ranker import BaseRanker, RankedResults
from rerankers import Document
from rerankers.results import Result
import json
import urllib.request
import urllib.error
from typing import Union, List, Optional, Tuple


class AzureReranker(BaseRanker):
    """
    Azure Cohere Reranker implementation using Azure AI Services.
    
    This class provides document reranking capabilities using Azure's Cohere reranker API.
    """
    
    def __init__(
        self, 
        model_name_or_path: str = "azure-reranker", 
        verbose: int = 1, 
        endpoint: str = None, 
        api_key: str = None, 
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize Azure Reranker
        
        Args:
            model_name_or_path: Name of the model (for identification)
            verbose: Verbosity level
            endpoint: Azure reranker endpoint URL
            api_key: Azure API key
            timeout: Request timeout in seconds
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_name_or_path, verbose)
        
        if not endpoint:
            raise ValueError("Azure reranker endpoint is required")
        if not api_key:
            raise ValueError("Azure API key is required")
            
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def score(self, query: str, doc: str) -> float:
        """
        Score a single document against a query using Azure reranker API
        
        Args:
            query: The search query
            doc: Single document to score (string)
            
        Returns:
            Relevance score as float
        """
        scores = self._call_azure_rerank(query, [doc])
        return scores[0] if scores else 0.0

    def rank(
        self, 
        query: str, 
        docs: Union[str, List[str], Document, List[Document]], 
        doc_ids: Optional[Union[List[str], List[int]]] = None, 
        **kwargs
    ) -> RankedResults:
        """
        Rank documents using Azure reranker API
        
        Args:
            query: The search query
            docs: Documents to rank (string, list of strings, Document, or list of Document objects)
            doc_ids: Optional document IDs
            **kwargs: Additional keyword arguments
            
        Returns:
            RankedResults object with ranked documents
        """
        # Normalize inputs
        normalized_docs, normalized_doc_ids = self._normalize_inputs(docs, doc_ids)
        
        # Extract text content for API call
        doc_texts = self._extract_document_texts(normalized_docs)
        
        # Call Azure reranker API
        scores = self._call_azure_rerank(query, doc_texts)
        
        # Create and sort results
        result_tuples = self._create_result_tuples(normalized_docs, scores, normalized_doc_ids)
        result_tuples.sort(key=lambda x: x[1], reverse=True)
        
        # Create Result objects for RankedResults
        results = []
        for i, (doc, score, doc_id) in enumerate(result_tuples):
            # Convert document to Document object if it's a string
            if isinstance(doc, str):
                document = Document(text=doc, doc_id=doc_id)
            else:
                document = doc
                
            # Create Result object with score and rank
            result = Result(
                document=document,
                score=score,
                rank=i + 1  # Rank starts from 1
            )
            results.append(result)
        
        # Return RankedResults with Result objects
        return RankedResults(
            results=results, 
            query=query,
            has_scores=True  # We have scores from Azure API
        )

    def _normalize_inputs(
        self, 
        docs: Union[str, List[str], Document, List[Document]], 
        doc_ids: Optional[Union[List[str], List[int]]]
    ) -> Tuple[List[Union[str, Document]], List[str]]:
        """
        Normalize input documents and doc_ids to consistent format
        
        Args:
            docs: Input documents
            doc_ids: Input document IDs
            
        Returns:
            Tuple of (normalized_docs, normalized_doc_ids)
        """
        # Convert single document to list
        if isinstance(docs, (str, Document)):
            docs = [docs]
            
        # Generate doc_ids if not provided
        if doc_ids is None:
            if docs and isinstance(docs[0], Document):
                doc_ids = [doc.doc_id for doc in docs]
            else:
                doc_ids = [str(i) for i in range(len(docs))]
        else:
            # Ensure doc_ids are strings
            doc_ids = [str(doc_id) for doc_id in doc_ids]
            
        return docs, doc_ids

    def _extract_document_texts(self, docs: List[Union[str, Document]]) -> List[str]:
        """
        Extract text content from documents
        
        Args:
            docs: List of documents (strings or Document objects)
            
        Returns:
            List of document texts
        """
        if docs and isinstance(docs[0], Document):
            return [doc.text for doc in docs]
        else:
            return docs

    def _create_result_tuples(
        self, 
        docs: List[Union[str, Document]], 
        scores: List[float], 
        doc_ids: List[str]
    ) -> List[Tuple[Union[str, Document], float, str]]:
        """
        Create result tuples from documents, scores, and doc_ids
        
        Args:
            docs: Original documents
            scores: Relevance scores
            doc_ids: Document IDs
            
        Returns:
            List of (document, score, doc_id) tuples
        """
        result_tuples = []
        for i, score in enumerate(scores):
            doc = docs[i] if i < len(docs) else ""
            doc_id = doc_ids[i] if i < len(doc_ids) else str(i)
            result_tuples.append((doc, score, doc_id))
        return result_tuples

    def _call_azure_rerank(self, query: str, docs: List[str]) -> List[float]:
        """
        Call Azure Cohere reranker API
        
        Args:
            query: Search query
            docs: List of documents to rank
            
        Returns:
            List of relevance scores
            
        Raises:
            RuntimeError: If API call fails or returns unexpected format
            ValueError: If API response format is invalid
        """
        if not docs:
            return []
            
        try:
            # Format data according to Azure Cohere rerank API
            data = {
                "query": query,
                "documents": docs,
                "top_n": len(docs)  # Return scores for all documents
            }
            
            body = json.dumps(data).encode('utf-8')
            
            headers = {
                'Content-Type': 'application/json', 
                'Accept': 'application/json', 
                'Authorization': f'Bearer {self.api_key}'
            }
            
            # Create request
            req = urllib.request.Request(self.endpoint, body, headers)
            
            # Make request with timeout
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = response.read()
                result_json = json.loads(result.decode('utf-8'))
                
                return self._parse_api_response(result_json, len(docs))
                    
        except urllib.error.HTTPError as error:
            error_msg = f"HTTP Error {error.code}: {error.reason}"
            try:
                error_detail = error.read().decode("utf8", 'ignore')
                error_msg += f"\nError details: {error_detail}"
            except Exception:
                pass
            raise RuntimeError(f"Failed to call Azure reranker API: {error_msg}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse API response as JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Azure reranker API: {str(e)}")

    def _parse_api_response(self, result_json: dict, num_docs: int) -> List[float]:
        """
        Parse Azure Cohere rerank API response
        
        Args:
            result_json: JSON response from API
            num_docs: Number of documents sent for ranking
            
        Returns:
            List of relevance scores in original document order
            
        Raises:
            ValueError: If response format is unexpected
        """
        if "results" not in result_json:
            raise ValueError(f"Unexpected API response format: missing 'results' field. Response: {result_json}")
            
        # Initialize scores array with zeros
        scores = [0.0] * num_docs
        
        # Parse results and place scores in correct positions
        for item in result_json["results"]:
            if not isinstance(item, dict):
                continue
                
            index = item.get("index")
            score = item.get("relevance_score", 0.0)
            
            if index is None or not isinstance(index, int):
                continue
                
            if 0 <= index < num_docs:
                scores[index] = float(score)
                
        return scores